mod circuit;
mod compression;
mod gate;
pub(crate) mod spread;
pub(crate) mod utils;
use std::convert::TryInto;

pub use compression::*;

use gate::ShaThreadBuilder;
use generic_array::GenericArray;
use halo2_base::gates::{RangeChip, RangeInstructions};
use halo2_base::halo2_proofs::plonk::Error;
use halo2_base::utils::BigPrimeField;
use halo2_base::QuantumCell;
use halo2_base::{gates::GateInstructions, AssignedValue};
use itertools::Itertools;
use sha2::compress256;
use spread::{SpreadChip, SpreadConfig};

// const Sha256BitChipRowPerRound: usize = 72;
// const BLOCK_SIZE: usize = 64;
// const DIGEST_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct AssignedHashResult<F: BigPrimeField> {
    pub input_len: AssignedValue<F>,
    pub input_bytes: Vec<AssignedValue<F>>,
    pub output_bytes: [AssignedValue<F>; 32],
}

pub struct Sha256Chip<'a, F: BigPrimeField> {
    spread: SpreadChip<'a, F>,
}

impl<'a, F: BigPrimeField> Sha256Chip<'a, F> {
    const ONE_ROUND_INPUT_BYTES: usize = 64;

    pub fn new(range: &'a RangeChip<F>) -> Self {
        Self {
            spread: SpreadChip::new(range),
        }
    }

    pub fn digest<const MAX_INPUT_SIZE: usize>(
        &self,
        thread_pool: &mut ShaThreadBuilder<F>,
        input: &'a [u8],
        // precomputed_input_len: Option<usize>,
        strict: bool,
    ) -> Result<AssignedHashResult<F>, Error> {
        let assigned_input = input
            .iter()
            .map(|byte| thread_pool.main().load_witness(F::from(*byte as u64)))
            .collect_vec();

        self.digest_assigned::<MAX_INPUT_SIZE>(
            thread_pool,
            assigned_input,
            None, //precomputed_input_len,
            strict,
        )
    }

    pub fn digest_assigned<const MAX_INPUT_SIZE: usize>(
        &self,
        thread_pool: &mut ShaThreadBuilder<F>,
        input: Vec<AssignedValue<F>>,
        precomputed_input_len: Option<usize>,
        strict: bool,
    ) -> Result<AssignedHashResult<F>, Error> {
        let input_byte_size = input.len();
        let precomputed_input_len = precomputed_input_len.unwrap_or(0);
        assert!(input_byte_size - precomputed_input_len <= MAX_INPUT_SIZE);
        let input_byte_size_with_9 = input_byte_size + 9;
        let one_round_size = Self::ONE_ROUND_INPUT_BYTES;

        let num_round = if input_byte_size_with_9 % one_round_size == 0 {
            input_byte_size_with_9 / one_round_size
        } else {
            input_byte_size_with_9 / one_round_size + 1
        };

        let mut assign_byte = |byte: u8| thread_pool.main().load_witness(F::from(byte as u64));

        let padded_size = one_round_size * num_round;
        let max_variable_byte_size = MAX_INPUT_SIZE;
        let max_variable_round = max_variable_byte_size / one_round_size;
        assert_eq!(precomputed_input_len % one_round_size, 0);
        assert!(padded_size - precomputed_input_len <= max_variable_byte_size);
        let zero_padding_byte_size = padded_size - input_byte_size_with_9;
        let remaining_byte_size = max_variable_byte_size + precomputed_input_len - padded_size;
        let precomputed_round = precomputed_input_len / one_round_size;
        assert_eq!(
            remaining_byte_size,
            one_round_size * (max_variable_round + precomputed_round - num_round)
        );
        let mut padded_inputs = input.to_vec();
        padded_inputs.push(assign_byte(0x80));

        for _ in 0..zero_padding_byte_size {
            padded_inputs.push(assign_byte(0u8));
        }
        let mut input_len_bytes = [0; 8];
        let le_size_bytes = (8 * input_byte_size).to_le_bytes();
        input_len_bytes[0..le_size_bytes.len()].copy_from_slice(&le_size_bytes);
        for byte in input_len_bytes.iter().rev() {
            padded_inputs.push(assign_byte(*byte));
        }

        assert_eq!(padded_inputs.len(), num_round * one_round_size);
        for _ in 0..remaining_byte_size {
            padded_inputs.push(assign_byte(0u8));
        }
        assert_eq!(
            padded_inputs.len(),
            max_variable_byte_size + precomputed_input_len
        );

        let range = self.spread.range();
        let gate = &range.gate;

        let assigned_input_byte_size = thread_pool
            .main()
            .load_witness(F::from(input_byte_size as u64));
        let assigned_num_round = thread_pool.main().load_witness(F::from(num_round as u64));
        let assigned_padded_size = gate.mul(
            thread_pool.main(),
            assigned_num_round,
            QuantumCell::Constant(F::from(one_round_size as u64)),
        );
        let assigned_input_with_9_size = gate.add(
            thread_pool.main(),
            assigned_input_byte_size,
            QuantumCell::Constant(F::from(9u64)),
        );
        let padding_size = gate.sub(
            thread_pool.main(),
            assigned_padded_size,
            assigned_input_with_9_size,
        );
        let padding_is_less_than_round =
            range.is_less_than_safe(thread_pool.main(), padding_size, one_round_size as u64);
        gate.assert_is_const(thread_pool.main(), &padding_is_less_than_round, &F::ONE);
        let assigned_precomputed_round = thread_pool
            .main()
            .load_witness(F::from(precomputed_round as u64));
        let assigned_target_round = gate.sub(
            thread_pool.main(),
            assigned_num_round,
            assigned_precomputed_round,
        );

        let assigned_num_round = thread_pool.main().load_witness(F::from(num_round as u64));

        // compute an initial state from the precomputed_input.
        let padded_bytes = padded_inputs
            .iter()
            .map(|byte| byte.value().get_lower_32() as u8)
            .collect_vec();
        let precomputed_input = &padded_bytes[0..precomputed_input_len];
        let mut last_state = INIT_STATE;
        let precomputed_blocks = precomputed_input
            .chunks(one_round_size)
            .map(GenericArray::clone_from_slice)
            .collect_vec();
        compress256(&mut last_state, &precomputed_blocks[..]);

        let mut assigned_last_state_vec = vec![last_state
            .iter()
            .map(|state| thread_pool.main().load_witness(F::from(*state as u64)))
            .collect_vec()];

        let assigned_input_bytes = padded_inputs;

        if strict {
            for &assigned in assigned_input_bytes.iter() {
                range.range_check(thread_pool.main(), assigned, 8);
            }
        }

        let mut num_processed_input = 0;
        while num_processed_input < MAX_INPUT_SIZE {
            let assigned_input_word_at_round =
                &assigned_input_bytes[num_processed_input..(num_processed_input + one_round_size)];
            let new_assigned_hs_out = sha256_compression(
                thread_pool,
                &self.spread,
                assigned_input_word_at_round,
                assigned_last_state_vec.last().unwrap(),
            )?;

            assigned_last_state_vec.push(new_assigned_hs_out);
            num_processed_input += one_round_size;
        }

        let zero = thread_pool.main().load_zero();
        let mut output_h_out = vec![zero; 8];
        for (n_round, assigned_state) in assigned_last_state_vec.into_iter().enumerate() {
            let selector = gate.is_equal(
                thread_pool.main(),
                QuantumCell::Constant(F::from(n_round as u64)),
                assigned_num_round,
            );
            for i in 0..8 {
                output_h_out[i] = gate.select(
                    thread_pool.main(),
                    assigned_state[i],
                    output_h_out[i],
                    selector,
                )
            }
        }
        let output_digest_bytes = output_h_out
            .into_iter()
            .flat_map(|assigned_word| {
                let be_bytes = assigned_word.value().get_lower_32().to_be_bytes().to_vec();
                let assigned_bytes = (0..4)
                    .map(|idx| {
                        let assigned = thread_pool
                            .main()
                            .load_witness(F::from(be_bytes[idx] as u64));
                        range.range_check(thread_pool.main(), assigned, 8);
                        assigned
                    })
                    .collect_vec();
                let mut sum = thread_pool.main().load_zero();
                for (idx, assigned_byte) in assigned_bytes.iter().copied().enumerate() {
                    sum = gate.mul_add(
                        thread_pool.main(),
                        assigned_byte,
                        QuantumCell::Constant(F::from(1u64 << (24 - 8 * idx))),
                        sum,
                    );
                }
                thread_pool.main().constrain_equal(&assigned_word, &sum);
                assigned_bytes
            })
            .collect_vec()
            .try_into()
            .unwrap();

        let result = AssignedHashResult {
            input_len: assigned_input_byte_size,
            input_bytes: assigned_input_bytes,
            output_bytes: output_digest_bytes,
        };
        Ok(result)
    }

    fn range(&self) -> &RangeChip<F> {
        self.spread.range()
    }
}

#[cfg(test)]
mod test {
    use crate::circuit::ShaCircuitBuilder;

    use super::*;
    use halo2_base::{
        halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr},
        utils::BigPrimeField,
    };

    use rand::{thread_rng, Rng};
    use sha2::{Digest, Sha256};

    const MAX_BYTE_SIZE1: usize = 128;
    const MAX_BYTE_SIZE2: usize = 128;

    fn test_circuit<F: BigPrimeField>(
        builder: &mut ShaCircuitBuilder<F>,
        input_vector: &[Vec<u8>],
        // precomputed_input_lens: Vec<usize>,
    ) -> Result<Vec<u8>, Error> {
        let mut output_bytes = vec![];
        let range = builder.range_chip(8);
        let sha256 = Sha256Chip::new(&range);

        let result0 = sha256.digest::<MAX_BYTE_SIZE1>(
            builder.core_mut(),
            &input_vector[0],
            // Some(precomputed_input_lens[0]),
            true,
        )?;
        output_bytes.append(
            &mut result0
                .output_bytes
                .into_iter()
                .map(|v| v.value().get_lower_32() as u8)
                .collect(),
        );
        let result1 = sha256.digest::<MAX_BYTE_SIZE2>(
            builder.core_mut(),
            &input_vector[1],
            // Some(precomputed_input_lens[1]),
            true,
        )?;
        output_bytes.append(
            &mut result1
                .output_bytes
                .into_iter()
                .map(|v| v.value().get_lower_32() as u8)
                .collect(),
        );

        builder.calculate_params(None);

        Ok(output_bytes)
    }

    #[test]
    fn test_sha256_correct1() {
        let k = 17;

        // Test vector: "abc"
        let test_input = vec!['a' as u8, 'b' as u8, 'c' as u8];

        let mut circuit = ShaCircuitBuilder::<Fr>::mock().use_k(k);

        let output_bytes = test_circuit(&mut circuit, &[test_input, vec![]]).unwrap();

        let test_output0: [u8; 32] = [
            0b10111010, 0b01111000, 0b00010110, 0b10111111, 0b10001111, 0b00000001, 0b11001111,
            0b11101010, 0b01000001, 0b01000001, 0b01000000, 0b11011110, 0b01011101, 0b10101110,
            0b00100010, 0b00100011, 0b10110000, 0b00000011, 0b01100001, 0b10100011, 0b10010110,
            0b00010111, 0b01111010, 0b10011100, 0b10110100, 0b00010000, 0b11111111, 0b01100001,
            0b11110010, 0b00000000, 0b00010101, 0b10101101,
        ];
        let test_output1 =
            hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap();
        let test_output = [test_output0.to_vec(), test_output1]
            .concat()
            .into_iter()
            // .map(|val| Fr::from_u128(val as u128))
            .collect_vec();

        let prover = MockProver::run(k as u32, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        assert_eq!(output_bytes, test_output);
    }

    #[test]
    fn test_sha256_correct2() {
        let k = 17;

        // Test vector: "0x0"
        let test_input = vec![0u8];

        let mut circuit = ShaCircuitBuilder::<Fr>::mock().use_k(k);

        let output_bytes = test_circuit(&mut circuit, &[test_input, vec![]]).unwrap();

        let test_output0 =
            hex::decode("6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d")
                .unwrap();
        let test_output1 =
            hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                .unwrap();
        let test_output = vec![test_output0, test_output1]
            .concat()
            .into_iter()
            // .map(|val| Fr::from_u128(val as u128))
            .collect_vec();

        let prover = MockProver::run(k as u32, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        assert_eq!(output_bytes, test_output);
    }

    #[test]
    fn test_sha256_correct3() {
        let k = 17;

        let test_inputs = vec![vec![0x1; 56], vec![0u8, 0u8, 0u8]];

        let mut circuit = ShaCircuitBuilder::<Fr>::mock().use_k(k);

        let output_bytes = test_circuit(&mut circuit, &test_inputs).unwrap();

        let test_output0 =
            hex::decode("51e14a913680f24c85fe3b0e2e5b57f7202f117bb214f8ffdd4ea0f4e921fd52")
                .unwrap();
        let test_output1 =
            hex::decode("709e80c88487a2411e1ee4dfb9f22a861492d20c4765150c0c794abd70f8147c")
                .unwrap();
        let test_output = vec![test_output0, test_output1]
            .concat()
            .into_iter()
            // .map(|val| Fr::from_u128(val as u128))
            .collect_vec();
        let prover = MockProver::run(k as u32, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        assert_eq!(output_bytes, test_output);
    }

    #[test]
    fn test_sha256_correct4() {
        let k = 17;

        fn gen_random_bytes(len: usize) -> Vec<u8> {
            let mut rng = thread_rng();
            (0..len).map(|_| rng.gen::<u8>()).collect()
        }
        let test_inputs = vec![gen_random_bytes(64), gen_random_bytes(64)];
        let test_output0 = Sha256::digest(&test_inputs[0]);
        let test_output1 = Sha256::digest(&test_inputs[1]);
        let mut builder = ShaThreadBuilder::<Fr>::mock();

        let mut circuit = ShaCircuitBuilder::<Fr>::mock();

        let output_bytes = test_circuit(&mut circuit, &test_inputs).unwrap();

        let test_output = vec![test_output0, test_output1]
            .concat()
            .into_iter()
            // .map(|val| Fr::from_u128(val as u128))
            .collect_vec();
        let prover = MockProver::run(k as u32, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        assert_eq!(output_bytes, test_output);
    }
}
