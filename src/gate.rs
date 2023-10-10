use halo2_base::{
    gates::{
        circuit::{CircuitBuilderStage, BaseCircuitParams},
        flex_gate::{threads::MultiPhaseCoreManager, BasicGateConfig, FlexGateConfigParams}, RangeChip,
    },
    halo2_proofs::circuit::{Region, Value},
    utils::BigPrimeField,
    virtual_region::{
        copy_constraints::{CopyConstraintManager, SharedCopyConstraintManager},
        manager::VirtualRegionManager,
    },
    Context, ContextCell,
};
use itertools::Itertools;

use super::SpreadConfig;

pub const FIRST_PHASE: usize = 0;

#[derive(Clone, Debug, Default)]
pub struct ShaThreadBuilder<F: BigPrimeField> {
    /// Threads for spread table assignment.
    pub threads_dense: Vec<Context<F>>,
    /// Threads for spread table assignment.
    pub threads_spread: Vec<Context<F>>,
    /// [`SinglePhaseCoreManager`] with threads for basic gate; also in charge of thread IDs
    pub core: MultiPhaseCoreManager<F>,
}

pub type ShaContexts<'a, F> = (&'a mut Context<F>, &'a mut Context<F>);

impl<F: BigPrimeField> ShaThreadBuilder<F> {
    pub fn new(witness_gen_only: bool) -> Self {
        Self {
            threads_spread: Vec::new(),
            threads_dense: Vec::new(),
            core: MultiPhaseCoreManager::new(witness_gen_only),
        }
    }

    pub fn mock() -> Self {
        Self::new(false)
    }

    pub fn keygen() -> Self {
        Self::new(false).unknown(true)
    }

    pub fn prover() -> Self {
        Self::new(true)
    }

    pub fn from_stage(stage: CircuitBuilderStage) -> Self {
        Self::new(stage == CircuitBuilderStage::Prover)
            .unknown(stage == CircuitBuilderStage::Keygen)
    }

    pub fn copy_manager(&self) -> SharedCopyConstraintManager<F> {
        self.core.copy_manager.clone()
    }

    /// Returns `self` with a given copy manager
    pub fn use_copy_manager(mut self, copy_manager: SharedCopyConstraintManager<F>) -> Self {
        self.core.set_copy_manager(copy_manager);
        self
    }

    pub fn unknown(mut self, use_unknown: bool) -> Self {
        self.core = self.core.unknown(use_unknown);
        self
    }

    pub fn main(&mut self) -> &mut Context<F> {
        self.core.main(FIRST_PHASE)
    }

    pub fn witness_gen_only(&self) -> bool {
        self.core.witness_gen_only()
    }

    pub fn use_unknown(&self) -> bool {
        self.core.use_unknown()
    }

    pub fn thread_count(&self) -> usize {
        self.core.phase_manager[0].thread_count()
    }

    pub fn get_new_thread_id(&mut self) -> usize {
        self.core.phase_manager[0].thread_count()
    }

    pub fn calculate_params(&self, k: usize, minimum_rows: Option<usize>) -> FlexGateConfigParams {
        self.core.calculate_params(k, minimum_rows)
    }
}

impl<F: BigPrimeField> VirtualRegionManager<F> for ShaThreadBuilder<F> {
    type Config = (Vec<BasicGateConfig<F>>, SpreadConfig<F>, usize); // usize = usable_rows

    fn assign_raw(&self, (gate, spread, usable_rows): &Self::Config, region: &mut Region<F>) {
        self.core.phase_manager[0].assign_raw(&(gate.clone(), *usable_rows), region);

        if self.core.witness_gen_only() {
            let mut copy_manager = self.core.copy_manager.lock().unwrap();

            assign_threads_sha(
                &self.threads_dense,
                &self.threads_spread,
                spread,
                region,
                self.use_unknown(),
                Some(&mut copy_manager),
            );
            // // in order to constrain equalities and assign constants, we copy the Spread/Dense equality constraints into the gate builder (it doesn't matter which context the equalities are in), so `GateThreadBuilder::assign_all` can take care of it
            // // the phase doesn't matter for equality constraints, so we use phase 0 since we're sure there's a main context there
            // let main_ctx = self.core.main();
            // for ctx in self
            //     .threads_spread
            //     .iter_mut()
            //     .chain(self.threads_dense.iter_mut())
            // {
            //     copy_manager
            //         .advice_equalities
            //         .append(&mut ctx.);
            //     main_ctx
            //         .constant_equality_constraints
            //         .append(&mut ctx.constant_equality_constraints);
            // }
        } else {
            assign_threads_sha(
                &self.threads_dense,
                &self.threads_spread,
                spread,
                region,
                false,
                None,
            );
        }
    }
}

impl<F: BigPrimeField> ShaThreadBuilder<F> {
    pub fn sha_contexts_pair(&mut self) -> (&mut Context<F>, ShaContexts<F>) {
        if self.threads_dense.is_empty() {
            self.new_thread_dense();
        }
        if self.threads_spread.is_empty() {
            self.new_thread_spread();
        }
        (
            self.core.main(FIRST_PHASE),
            (
                self.threads_dense.last_mut().unwrap(),
                self.threads_spread.last_mut().unwrap(),
            ),
        )
    }

    pub fn new_thread_dense(&mut self) -> &mut Context<F> {
        let thread_id = self.get_new_thread_id();
        self.threads_dense.push(Context::new(
            self.witness_gen_only(),
            FIRST_PHASE,
            self.core.phase_manager[0].type_of(),
            thread_id,
            self.core.copy_manager.clone(),
        ));
        self.threads_dense.last_mut().unwrap()
    }

    pub fn new_thread_spread(&mut self) -> &mut Context<F> {
        let thread_id = self.get_new_thread_id();
        self.threads_spread.push(Context::new(
            self.witness_gen_only(),
            FIRST_PHASE,
            self.core.phase_manager[0].type_of(),
            thread_id,
            self.core.copy_manager.clone(),
        ));
        self.threads_spread.last_mut().unwrap()
    }
}

/// Pure advice witness assignment in a single phase. Uses preprocessed `break_points` to determine when
/// to split a thread into a new column.
#[allow(clippy::type_complexity)]
pub fn assign_threads_sha<F: BigPrimeField>(
    threads_dense: &[Context<F>],
    threads_spread: &[Context<F>],
    spread: &SpreadConfig<F>,
    region: &mut Region<F>,
    use_unknown: bool,
    mut copy_manager: Option<&mut CopyConstraintManager<F>>,
) {
    let mut num_limb_sum = 0;
    let mut row_offset = 0;
    for (ctx_dense, ctx_spread) in threads_dense.iter().zip_eq(threads_spread.iter()) {
        for (i, (&advice_dense, &advice_spread)) in ctx_dense
            .advice
            .iter()
            .zip_eq(ctx_spread.advice.iter())
            .enumerate()
        {
            let column_idx = num_limb_sum % spread.num_advice_columns;
            let value_dense = if use_unknown {
                Value::unknown()
            } else {
                Value::known(advice_dense)
            };

            let cell_dense = region
                .assign_advice(
                    || "dense",
                    spread.denses[column_idx],
                    row_offset,
                    || value_dense,
                )
                .unwrap()
                .cell();

            if let Some(copy_manager) = copy_manager.as_mut() {
                copy_manager.assigned_advices.insert(
                    ContextCell::new(ctx_dense.type_id(), ctx_dense.id(), i),
                    cell_dense,
                );
            }

            let value_spread = if use_unknown {
                Value::unknown()
            } else {
                Value::known(advice_spread)
            };

            let cell_spread = region
                .assign_advice(
                    || "spread",
                    spread.spreads[column_idx],
                    row_offset,
                    || value_spread,
                )
                .unwrap()
                .cell();

            if let Some(copy_manager) = copy_manager.as_mut() {
                copy_manager.assigned_advices.insert(
                    ContextCell::new(ctx_spread.type_id(), ctx_spread.id(), i),
                    cell_spread,
                );
            }

            num_limb_sum += 1;
            if column_idx == spread.num_advice_columns - 1 {
                row_offset += 1;
            }
            row_offset += 1;
        }
    }
}
