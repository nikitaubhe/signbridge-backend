import numpy as np

def get_finger_states(landmarks):
    """
    Analyzes 21 landmarks to determine if fingers are EXTENDED.
    """
    pts = landmarks.reshape(21, 3)
    wrist = pts[0]
    
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    mcps = [5, 9, 13, 17]
    
    up_states = []
    
    # ── Thumb Check (More robust) ──
    # Thumb is complex. Check if tip is far from both wrist and pinky base (17)
    dist_tip_wrist = np.linalg.norm(pts[4] - pts[0])
    dist_ip_wrist  = np.linalg.norm(pts[3] - pts[0])
    # Also check if thumb is "out" sideways
    up_states.append(dist_tip_wrist > dist_ip_wrist * 1.1)
    
    # ── Other Fingers ──
    for t, p, m in zip(tips, pips, mcps):
        d_tip = np.linalg.norm(pts[t] - wrist)
        d_pip = np.linalg.norm(pts[p] - wrist)
        d_mcp = np.linalg.norm(pts[m] - wrist)
        
        # A finger is "Up" if tip is further than PIP, and PIP is further than MCP
        up_states.append(d_tip > d_pip and d_pip > d_mcp * 1.1)
        
    return up_states

def detect_sign_heuristic(landmarks):
    if np.all(landmarks == 0): return None, 0.0
    
    pts = landmarks.reshape(21, 3)
    states = get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = states
    
    # ── Specialized Checks ──
    
    # 1. STOP / FIST (All fingers very close to palm)
    # Average distance of all tips to wrist
    tip_indices = [4, 8, 12, 16, 20]
    avg_dist = np.mean([np.linalg.norm(pts[i] - pts[0]) for i in tip_indices])
    # If average tip distance is small, it's a fist
    if avg_dist < 0.25: # Heuristic threshold for "compactness"
        return 'STOP', 0.95

    # 2. PEACE (V Sign)
    # Index and Middle up, others down. 
    # CRITICAL: Check distance between Index Tip and Middle Tip (must be separated)
    if index and middle and not ring and not pinky:
        dist_v = np.linalg.norm(pts[8] - pts[12])
        if dist_v > 0.08: # They must be spread apart
            return 'PEACE', 0.98

    # 3. HELLO (Open Palm)
    if sum(states[1:]) >= 3: # Most fingers up
        return 'HELLO', 0.95
    
    # 4. YES (Thumb Up)
    if thumb and not any(states[1:]):
        # Thumb tip must be the highest point (lowest Y)
        if pts[4][1] < pts[5][1]: 
            return 'YES', 0.95
        
    # 5. NO (Thumb Down)
    if not any(states[1:]) and pts[4][1] > pts[0][1] + 0.1:
        return 'NO', 0.95
        
    # 6. LOVE (I Love You)
    if thumb and index and pinky and not middle and not ring:
        return 'LOVE', 0.98
        
    # 7. OKAY (OK Sign)
    dist_ok = np.linalg.norm(pts[4] - pts[8])
    if dist_ok < 0.05:
        return 'OKAY', 0.98
        
    # 8. THANK YOU (Pointing)
    if index and not any([middle, ring, pinky]):
        return 'THANK_YOU', 0.95
        
    return None, 0.0
