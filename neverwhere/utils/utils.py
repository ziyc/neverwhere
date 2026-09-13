import numpy as np
import torch
from dm_control import mujoco

ISAAC_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

JOINT_IDX_MAPPING = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

UNITREE_DOF_NAMES = np.array([ISAAC_DOF_NAMES])[:, JOINT_IDX_MAPPING][0].tolist()

def euler_from_quaternion_np(quat_angle):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw) using NumPy.
    roll is rotation around x in radians (counterclockwise),
    pitch is rotation around y in radians (counterclockwise),
    yaw is rotation around z in radians (counterclockwise).
    """
    x, y, z, w = quat_angle

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def quat_from_euler_xyz_np(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=-1)

def smart_delta_yaw(yaw1, yaw2):
    delta = yaw2 - yaw1

    # Adjust differences to find the shortest path
    delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi

    return delta

def quat_rotate_inverse_np(q, v):
    # Assuming q is of shape (4,) and v is of shape (3,)
    q_w = q[-1]  # Scalar
    q_vec = q[:3]  # Vector part of the quaternion (3,)

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0

    # Combine the components
    result = a - b + c
    return result

def get_geom_speed(model, data, geom_name):
    """Returns the angular velocity of a geom."""
    geom_vel = np.zeros(6, dtype=np.float64)
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    geom_id = data.geom(geom_name).id
    mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
    return geom_vel[:3]

def normalize_np(quat):
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return quat / norm

def quat_apply_np(quat, vec):
    # Ensure quat is in the shape (-1, 4) and vec is in the shape (-1, 3)
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)

    # Separate the quaternion into xyz (vector part) and w (scalar part)
    xyz = quat[:, :3]
    w = quat[:, 3:].reshape(-1, 1)

    # Compute the cross product t = 2 * cross(xyz, vec)
    t = np.cross(xyz, vec) * 2

    # Compute the rotated vector
    # vec + w*t + cross(xyz, t)
    rotated_vec = vec + w * t + np.cross(xyz, t)

    # Return the rotated vector, reshaped back to the original vec shape
    return rotated_vec.reshape(vec.shape)

def quat_apply_yaw_np(quat, vec):
    quat_yaw = np.copy(quat).reshape(-1, 4)
    quat_yaw[:, :2] = 0.0  # Set the first three columns to zero, assuming w last
    quat_yaw = normalize_np(quat_yaw)  # Normalize the quaternion
    return quat_apply_np(quat_yaw, vec)  # Apply the quaternion rotation to the vector
