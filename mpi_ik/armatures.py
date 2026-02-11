"""Armature definitions for MANO, SMPL, and SMPL-H models.

Each armature class defines the skeleton topology: number of joints,
extended keypoint vertex indices, and human-readable joint labels.
"""

from __future__ import annotations


class MANOArmature:
    """Armature for the MANO hand model (16 joints + 5 fingertip keypoints)."""

    n_joints: int = 16

    keypoints_ext: list[int] = [333, 444, 672, 555, 744]

    n_keypoints: int = n_joints + len(keypoints_ext)

    labels: list[str] = [
        'W',                        # 0
        'I0', 'I1', 'I2',           # 1-3
        'M0', 'M1', 'M2',           # 4-6
        'L0', 'L1', 'L2',           # 7-9
        'R0', 'R1', 'R2',           # 10-12
        'T0', 'T1', 'T2',           # 13-15
        # extended
        'I3', 'M3', 'L3', 'R3', 'T3',  # 16-20
    ]


class SMPLArmature:
    """Armature for the SMPL body model (24 joints + 5 extended keypoints)."""

    n_joints: int = 24

    keypoints_ext: list[int] = [2446, 5907, 3216, 6618, 411]

    n_keypoints: int = n_joints + len(keypoints_ext)

    labels: list[str] = [
        'pelvis',
        'left leg root', 'right leg root',
        'lowerback',
        'left knee', 'right knee',
        'upperback',
        'left ankle', 'right ankle',
        'thorax',
        'left toes', 'right toes',
        'lowerneck',
        'left clavicle', 'right clavicle',
        'upperneck',
        'left armroot', 'right armroot',
        'left elbow', 'right elbow',
        'left wrist', 'right wrist',
        'left hand', 'right hand',
        # extended
        'left finger tip', 'right finger tip',
        'left toe tip', 'right toe tip',
        'head top',
    ]


class SMPLHArmature:
    """Armature for the SMPL-H body+hand model (52 joints + 13 extended keypoints)."""

    n_joints: int = 52

    keypoints_ext: list[int] = [
        2746, 2320, 2446, 2557, 2674,
        6191, 5781, 5907, 6018, 6135,
        3216, 6618, 411,
    ]

    n_keypoints: int = n_joints + len(keypoints_ext)

    labels: list[str] = [
        'pelvis',
        'left leg root', 'right leg root',
        'lowerback',
        'left knee', 'right knee',
        'upperback',
        'left ankle', 'right ankle',
        'thorax',
        'left toes', 'right toes',
        'lowerneck',
        'left clavicle', 'right clavicle',
        'upperneck',
        'left armroot', 'right armroot',
        'left elbow', 'right elbow',
        'left wrist', 'right wrist',
        'left hand', 'right hand',
        # left hand joints (indices 24-37, MANO finger convention)
        'L_I0', 'L_I1', 'L_I2',
        'L_M0', 'L_M1', 'L_M2',
        'L_L0', 'L_L1', 'L_L2',
        'L_R0', 'L_R1', 'L_R2',
        'L_T0', 'L_T1',
        # right hand joints (indices 38-51, MANO finger convention)
        'R_I0', 'R_I1', 'R_I2',
        'R_M0', 'R_M1', 'R_M2',
        'R_L0', 'R_L1', 'R_L2',
        'R_R0', 'R_R1', 'R_R2',
        'R_T0', 'R_T1',
        # extended keypoints
        'left thumb', 'left index', 'left middle', 'left ring', 'left little',
        'right thumb', 'right index', 'right middle', 'right ring', 'right little',
        'left toe tip', 'right toe tip', 'head top',
    ]
