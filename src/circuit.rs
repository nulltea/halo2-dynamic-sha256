use halo2_base::{
    gates::{
        circuit::{BaseCircuitParams, BaseConfig, CircuitBuilderStage, MaybeRangeConfig},
        flex_gate::{
            threads::SinglePhaseCoreManager, FlexGateConfigParams, MultiPhaseThreadBreakPoints,
        },
        RangeChip,
    },
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        plonk::{Circuit, ConstraintSystem, Error},
    },
    utils::BigPrimeField,
    virtual_region::{lookups::LookupAnyManager, manager::VirtualRegionManager},
    AssignedValue, Context,
};

use crate::{gate::ShaThreadBuilder, spread::SpreadConfig};

const MAX_PHASE: usize = 3;

#[derive(Debug, Clone)]
pub struct SHAConfig<F: BigPrimeField> {
    pub compression: SpreadConfig<F>,
    pub base: BaseConfig<F>,
}

impl<F: BigPrimeField> SHAConfig<F> {
    pub fn configure(meta: &mut ConstraintSystem<F>, params: BaseCircuitParams) -> Self {
        let degree = params.k;
        let mut base = BaseConfig::configure(meta, params);
        let compression = SpreadConfig::configure(meta, 8, 1);

        // base.gate.max_rows = (1 << degree) - meta.minimum_rows();
        Self { base, compression }
    }
}

pub struct ShaCircuitBuilder<F: BigPrimeField> {
    // pub builder: RefCell<ShaThreadBuilder<F>>,
    pub core: ShaThreadBuilder<F>,
    // pub break_points: RefCell<MultiPhaseThreadBreakPoints>, // `RefCell` allows the circuit to record break points in a keygen call of `synthesize` for use in later witness gen
    /// The range lookup manager
    pub(super) lookup_manager: [LookupAnyManager<F, 1>; MAX_PHASE],
    /// Configuration parameters for the circuit shape
    pub config_params: BaseCircuitParams,
    /// The assigned instances to expose publicly at the end of circuit synthesis
    pub assigned_instances: Vec<Vec<AssignedValue<F>>>,
}

impl<F: BigPrimeField> ShaCircuitBuilder<F> {
    pub fn new(witness_gen_only: bool) -> Self {
        let core = ShaThreadBuilder::new(witness_gen_only);
        let lookup_manager =
            [(); MAX_PHASE].map(|_| LookupAnyManager::new(witness_gen_only, core.copy_manager()));
        Self {
            core,
            lookup_manager,
            assigned_instances: vec![],
            config_params: BaseCircuitParams::default(),
        }
    }

    pub fn from_stage(stage: CircuitBuilderStage) -> Self {
        Self::new(stage == CircuitBuilderStage::Prover)
            .unknown(stage == CircuitBuilderStage::Keygen)
    }

    pub fn unknown(mut self, use_unknown: bool) -> Self {
        self.core = self.core.unknown(use_unknown);
        self
    }

    /// Creates a new [ShaCircuitBuilder] with `use_unknown` of [ShaThreadBuilder] set to true.
    pub fn keygen() -> Self {
        Self::from_stage(CircuitBuilderStage::Keygen)
    }

    /// Creates a new [ShaCircuitBuilder] with `use_unknown` of [GateThreadBuilder] set to false.
    pub fn mock() -> Self {
        Self::from_stage(CircuitBuilderStage::Mock)
    }

    /// Creates a new [ShaCircuitBuilder].
    pub fn prover() -> Self {
        Self::from_stage(CircuitBuilderStage::Prover)
    }

    /// The log_2 size of the lookup table, if using.
    pub fn lookup_bits(&self) -> Option<usize> {
        self.config_params.lookup_bits
    }

    /// Set lookup bits
    pub fn set_lookup_bits(&mut self, lookup_bits: usize) {
        self.config_params.lookup_bits = Some(lookup_bits);
    }

    /// Returns new with lookup bits
    pub fn use_lookup_bits(mut self, lookup_bits: usize) -> Self {
        self.set_lookup_bits(lookup_bits);
        self
    }

    /// Sets new `k` = log2 of domain
    pub fn set_k(&mut self, k: usize) {
        self.config_params.k = k;
    }

    /// Returns new with `k` set
    pub fn use_k(mut self, k: usize) -> Self {
        self.set_k(k);
        self
    }

    /// Set config params
    pub fn set_params(&mut self, params: BaseCircuitParams) {
        self.config_params = params;
    }

    /// Returns new with config params
    pub fn use_params(mut self, params: BaseCircuitParams) -> Self {
        self.set_params(params);
        self
    }

    pub fn core_mut(&mut self) -> &mut ShaThreadBuilder<F> {
        &mut self.core
    }

    /// Returns a mutable reference to the [Context] of a gate thread. Spawns a new thread for the given phase, if none exists.
    /// * `phase`: The challenge phase (as an index) of the gate thread.
    pub fn main(&mut self) -> &mut Context<F> {
        self.core.main()
    }

    /// Returns [SinglePhaseCoreManager] with the virtual region with all core threads in the given phase.
    pub fn pool(&mut self, phase: usize) -> &mut SinglePhaseCoreManager<F> {
        self.core.core.phase_manager.get_mut(phase).unwrap()
    }

    fn total_lookup_advice_per_phase(&self) -> Vec<usize> {
        self.lookup_manager
            .iter()
            .map(|lm| lm.total_rows())
            .collect()
    }

    pub fn calculate_params(&mut self, minimum_rows: Option<usize>) -> BaseCircuitParams {
        let k = self.config_params.k;
        let ni = self.config_params.num_instance_columns;
        let max_rows = (1 << k) - minimum_rows.unwrap_or(0);

        // clone everything so we don't alter the circuit in any way for later calls
        let gate_params = self.core.clone().calculate_params(k, minimum_rows);

        let total_lookup_advice_per_phase = self.total_lookup_advice_per_phase();
        let num_lookup_advice_per_phase = total_lookup_advice_per_phase
            .iter()
            .map(|count| (count + max_rows - 1) / max_rows)
            .collect::<Vec<_>>();

        let params = BaseCircuitParams {
            k: gate_params.k,
            num_advice_per_phase: gate_params.num_advice_per_phase,
            num_fixed: gate_params.num_fixed,
            num_lookup_advice_per_phase,
            lookup_bits: self.lookup_bits(),
            num_instance_columns: ni,
        };

        self.config_params = params.clone();
        #[cfg(feature = "display")]
        {
            println!("Total range check advice cells to lookup per phase: {total_lookup_advice_per_phase:?}");
            log::info!("Auto-calculated config params:\n {params:#?}");
        }
        params
    }

    pub fn range_chip(&self, lookup_bits: usize) -> RangeChip<F> {
        RangeChip::new(lookup_bits, self.lookup_manager.clone())
    }

    // re-usable function for synthesize
    #[allow(clippy::type_complexity)]
    pub fn sub_synthesize(
        &self,
        config: &SHAConfig<F>,
        layouter: &mut impl Layouter<F>,
    ) -> Result<(), Error> {
        if let MaybeRangeConfig::WithRange(config) = &config.base.base {
            config
                .load_lookup_table(layouter)
                .expect("load lookup table should not fail");
        }

        // let mut first_pass = SKIP_FIRST_PASS;
        // let witness_gen_only = self.builder.borrow().witness_gen_only();

        // let mut assigned_advices = HashMap::new();

        config.compression.load(layouter)?;

        layouter.assign_region(
            || "ShaCircuitBuilder generated circuit",
            |mut region| {
                // if first_pass {
                //     first_pass = false;
                //     return Ok(());
                // }

                let usable_rows = config.base.gate().max_rows;
                self.core.assign_raw(
                    &(
                        config.base.gate().basic_gates[0].clone(),
                        config.compression.clone(),
                        usable_rows,
                    ),
                    &mut region,
                );

                // Only assign cells to lookup if we're sure we're doing range lookups
                if let MaybeRangeConfig::WithRange(config) = &config.base {
                    self.assign_lookups_in_phase(config, &mut region, 0);
                }

                // Impose equality constraints
                if !self.core.witness_gen_only() {
                    self.core
                        .copy_manager()
                        .assign_raw(config.base.constants(), &mut region);
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

impl<F: BigPrimeField> Circuit<F> for ShaCircuitBuilder<F> {
    type Config = SHAConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = BaseCircuitParams;

    fn params(&self) -> Self::Params {
        self.config_params.clone()
    }

    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }

    fn configure_with_params(meta: &mut ConstraintSystem<F>, params: Self::Params) -> Self::Config {
        SHAConfig::configure(meta, params)
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> SHAConfig<F> {
        unreachable!("You must use configure_with_params");
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        self.sub_synthesize(&config, &mut layouter)?;
        Ok(())
    }
}
