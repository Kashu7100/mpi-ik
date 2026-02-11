"""Tests for armature definitions."""

from mpi_ik.armatures import MANOArmature, SMPLArmature, SMPLHArmature

ARMATURES = [MANOArmature, SMPLArmature, SMPLHArmature]


def test_label_count_matches_n_keypoints():
    for arm in ARMATURES:
        assert len(arm.labels) == arm.n_keypoints, (
            f'{arm.__name__}: len(labels)={len(arm.labels)} != n_keypoints={arm.n_keypoints}'
        )


def test_n_keypoints_equals_n_joints_plus_ext():
    for arm in ARMATURES:
        assert arm.n_keypoints == arm.n_joints + len(arm.keypoints_ext), (
            f'{arm.__name__}: n_keypoints mismatch'
        )


def test_no_suspiciously_long_labels():
    """Catch missing-comma bugs that silently concatenate adjacent strings."""
    max_label_len = 25
    for arm in ARMATURES:
        for label in arm.labels:
            assert len(label) <= max_label_len, (
                f'{arm.__name__}: label {label!r} is suspiciously long '
                f'({len(label)} chars) - possible missing comma'
            )


def test_mano_specific_counts():
    assert MANOArmature.n_joints == 16
    assert MANOArmature.n_keypoints == 21


def test_smpl_specific_counts():
    assert SMPLArmature.n_joints == 24
    assert SMPLArmature.n_keypoints == 29


def test_smplh_specific_counts():
    assert SMPLHArmature.n_joints == 52
    assert SMPLHArmature.n_keypoints == 65
