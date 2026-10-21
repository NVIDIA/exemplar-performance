// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pacing mode names shared by the replayers, the profilers and run manifests.

/// Replay the recorded work back to back, without the recorded gaps.
pub const UNPACED: &str = "unpaced";

/// Former names, still accepted on input; reports always carry the current name.
const ALIASES: &[(&str, &str)] = &[("as-fast-as-possible", UNPACED)];

/// The current name for a pacing mode given by its current or former name.
pub fn canonical(value: &str) -> &str {
    ALIASES
        .iter()
        .find(|(former, _)| *former == value)
        .map_or(value, |(_, current)| current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn former_name_maps_to_unpaced_and_others_pass_through() {
        assert_eq!(canonical("as-fast-as-possible"), "unpaced");
        assert_eq!(canonical("unpaced"), "unpaced");
        assert_eq!(canonical("recorded"), "recorded");
        assert_eq!(canonical("llm-gaps"), "llm-gaps");
    }
}
