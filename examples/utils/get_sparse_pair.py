import numpy as np
import math

def read_images_txt(images_file):
    """
    Reads COLMAP images.txt, skipping 2D feature lines.
    Returns {image_id: {'q': [qw,qx,qy,qz], 't':[tx,ty,tz], 'camera_id':..., 'name':...} }
    """
    images_data = {}
    with open(images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue

            # Try parse image_id as int:
            try:
                image_id = int(parts[0])
            except ValueError:
                continue  # 2D feature line
            
            # Parse the camera pose
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
            except ValueError:
                continue

            name = " ".join(parts[9:])
            images_data[image_id] = {
                'q': np.array([qw, qx, qy, qz], dtype=float),
                't': np.array([tx, ty, tz], dtype=float),
                'camera_id': camera_id,
                'name': name
            }
    return images_data

def quaternion_to_rotation_matrix(q):
    """
    Convert [qw, qx, qy, qz] to 3x3 rotation matrix (COLMAP's convention).
    """
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx**2 + qy**2)]
    ], dtype=float)
    return R

def get_camera_forward_direction(q):
    """
    COLMAP cameras look along -Z in local coords.
    So the forward direction in world coords is R^T * [0, 0, -1].
    """
    R = quaternion_to_rotation_matrix(q)
    forward_cam = np.array([0, 0, -1], dtype=float)
    forward_world = R.T @ forward_cam
    return forward_world / np.linalg.norm(forward_world)

def angle_between_vectors(v1, v2):
    """
    Returns angle in degrees between two 3D vectors v1 and v2.
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    dot_val = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return math.degrees(math.acos(dot_val))

def main():
    images_file = "path/to/sparse/0/images.txt"

    # 1) Reference image:
    ref_id = 1  # e.g., the ID for 0000.png
    # 2) Desired angle for A-C:
    target_angle_AC = 45.0  # We want angle(A, C) ~ 45 degrees
    # 3) Tolerance for angle sum check:
    #    angle(A,B) + angle(B,C) ~ angle(A,C)
    angle_sum_tolerance = 5.0  # degrees

    images_data = read_images_txt(images_file)

    # Build forward directions
    forward_dirs = {}
    for img_id, info in images_data.items():
        forward_dirs[img_id] = get_camera_forward_direction(info['q'])

    # Make sure reference is in dataset
    if ref_id not in images_data:
        raise ValueError(f"Reference ID={ref_id} not found")

    # A's forward direction
    fA = forward_dirs[ref_id]

    # We'll gather all other IDs
    all_ids = sorted(images_data.keys())
    others = [x for x in all_ids if x != ref_id]

    results = []
    for iB_idx in range(len(others)):
        for iC_idx in range(iB_idx+1, len(others)):
            B_id = others[iB_idx]
            C_id = others[iC_idx]
            fB = forward_dirs[B_id]
            fC = forward_dirs[C_id]

            # angles:
            AB = angle_between_vectors(fA, fB)  # angle(A,B)
            AC = angle_between_vectors(fA, fC)  # angle(A,C)
            BC = angle_between_vectors(fB, fC)  # angle(B,C)

            # 1) Check if AB + BC ~ AC
            #    We'll measure difference in sum
            sum_diff = abs((AB + BC) - AC)

            if sum_diff <= angle_sum_tolerance:
                # Then also check how close AC is to target
                angle_diff = abs(AC - target_angle_AC)
                results.append((B_id, C_id, AB, BC, AC, sum_diff, angle_diff))

    # Sort by how close AC is to the target angle
    # If you also want to break ties by sum_diff, you can do that secondarily
    results.sort(key=lambda x: x[6])  # x[6] = angle_diff

    print(f"Found {len(results)} pairs (B, C) s.t. A={ref_id}, sum_of_angles ~ AC, and AC near {target_angle_AC} deg.")
    max_show = 20
    for idx, (b_id, c_id, AB, BC, AC, sum_diff, angle_diff) in enumerate(results[:max_show], 1):
        print(f"[{idx}] B={b_id} ({images_data[b_id]['name']}), C={c_id} ({images_data[c_id]['name']})")
        print(f"   angle(A,B)={AB:.2f}, angle(B,C)={BC:.2f}, angle(A,C)={AC:.2f}")
        print(f"   AB+BC-AC = {AB+BC-AC:.2f} (sum_diff={sum_diff:.2f}), AC_target_diff={angle_diff:.2f}")

    if len(results) > max_show:
        print(f"(showing only first {max_show} of {len(results)})")

if __name__ == "__main__":
    main()
