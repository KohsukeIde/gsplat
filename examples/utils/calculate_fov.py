import os
import glob
import cv2
import numpy as np

def main():
    #------------------------------------------------------------
    # 1. パラメータ・準備
    #------------------------------------------------------------
    reference_image_path = "/groups/gag51404/ide/data/DTU/scan63/images/0023.png"
    image_folder = os.path.dirname(reference_image_path)
    
    feature_detector = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    fx = fy = 1000.0
    cx = cy = 500.0
    K = np.array([[fx,   0, cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float64)

    #------------------------------------------------------------
    # 2. 基準画像を読み込み、特徴点抽出
    #------------------------------------------------------------
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"基準画像が読み込めません: {reference_image_path}")
        return
    ref_keypoints, ref_descriptors = feature_detector.detectAndCompute(ref_img, None)

    #------------------------------------------------------------
    # 3. フォルダ内の他画像パスを取得(ソートする)
    #------------------------------------------------------------
    image_paths = glob.glob(os.path.join(image_folder, "*.*"))
    # アルファベット順にソート
    image_paths = sorted(image_paths)
    #------------------------------------------------------------

    for img_path in image_paths:
        # 基準画像はスキップ
        if img_path == reference_image_path:
            continue

        # 画像読み込み
        target_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if target_img is None:
            continue

        # 特徴点抽出
        tgt_keypoints, tgt_descriptors = feature_detector.detectAndCompute(target_img, None)
        if tgt_descriptors is None:
            print(f"特徴量が検出できませんでした: {img_path}")
            continue

        # 特徴点マッチング
        knn_matches = matcher.knnMatch(ref_descriptors, tgt_descriptors, k=2)
        good_matches = []
        ratio_thresh = 0.7  # Loweのratio test
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print(f"マッチ数が少ないためスキップ: {img_path}")
            continue

        #--------------------------------------------------------
        # エッセンシャル行列からR, tを復元
        #--------------------------------------------------------
        ref_pts = np.array([ref_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        tgt_pts = np.array([tgt_keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float64)

        E, mask = cv2.findEssentialMat(
            ref_pts, tgt_pts, 
            cameraMatrix=K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        if E is None or E.shape[0] < 3:
            print(f"エッセンシャル行列の推定に失敗: {img_path}")
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, ref_pts, tgt_pts, cameraMatrix=K)

        # 回転角度を計算
        trace_val = np.trace(R)
        val = max(min((trace_val - 1.0)/2.0, 1.0), -1.0)  # arccosの入力クランプ
        angle_rad = np.arccos(val)
        angle_deg = np.degrees(angle_rad)

        # 並進方向（tは単位ベクトル）
        t_norm = t / (np.linalg.norm(t) + 1e-8)

        print("-------------------------------------------------")
        print(f"対象画像: {img_path}")
        print(f"  マッチ数: {len(good_matches)}")
        print(f"  回転角度: {angle_deg:.2f} 度")
        print(f"  並進ベクトル(方向): {t_norm.ravel()}")

if __name__ == "__main__":
    main()
