import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def extract_maze_structure(video_path, roi, frame_number=0, maze_threshold=160):
    """
    Extract maze structure and save as JSON
    
    Parameters:
    -----------
    video_path : str
        Path to video file
    roi : tuple
        Region of interest as (x, y, width, height)
    frame_number : int
        Frame to extract maze from (default: 0)
    maze_threshold : int
        Threshold for detecting white maze floor
    """
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR: Could not read frame")
        return None
    
    # Apply ROI
    x, y, w, h = roi
    frame_roi = frame[y:y+h, x:x+w].copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create maze mask (white areas)
    _, maze_mask = cv2.threshold(blurred, maze_threshold, 255, cv2.THRESH_BINARY)
    
    # Invert to get walls (dark areas)
    walls_mask = cv2.bitwise_not(maze_mask)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    walls_mask = cv2.morphologyEx(walls_mask, cv2.MORPH_CLOSE, kernel)
    maze_mask = cv2.morphologyEx(maze_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of maze compartments (white areas)
    compartment_contours, _ = cv2.findContours(maze_mask, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours of walls (dark areas)
    wall_contours, _ = cv2.findContours(walls_mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    # Create maze structure dictionary
    maze_structure = {
        'metadata': {
            'video_path': video_path,
            'frame_number': frame_number,
            'roi': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'maze_threshold': maze_threshold,
            'roi_dimensions': {
                'width': int(w),
                'height': int(h)
            }
        },
        'compartments': [],
        'walls': [],
        'statistics': {}
    }
    
    # Process compartments (navigable areas)
    print("\nProcessing compartments (white maze areas)...")
    for i, contour in enumerate(compartment_contours):
        area = cv2.contourArea(contour)
        
        # Only include significant compartments (not tiny noise)
        if area > 500:  # Adjust threshold as needed
            # Get bounding box
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x_c + w_c//2, y_c + h_c//2
            
            # Simplify contour for smaller JSON
            epsilon = 0.01 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            compartment_data = {
                'id': i,
                'area': float(area),
                'centroid': {'x': int(cx), 'y': int(cy)},
                'bounding_box': {
                    'x': int(x_c),
                    'y': int(y_c),
                    'width': int(w_c),
                    'height': int(h_c)
                },
                'contour': simplified_contour.squeeze().tolist()  # Convert to list for JSON
            }
            
            maze_structure['compartments'].append(compartment_data)
            print(f"  Compartment {i}: Area={area:.0f} pixels, Center=({cx}, {cy})")
    
    # Process walls
    print("\nProcessing walls (dark areas)...")
    for i, contour in enumerate(wall_contours):
        area = cv2.contourArea(contour)
        
        # Only include significant walls
        if area > 200:
            x_w, y_w, w_w, h_w = cv2.boundingRect(contour)
            
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            wall_data = {
                'id': i,
                'area': float(area),
                'bounding_box': {
                    'x': int(x_w),
                    'y': int(y_w),
                    'width': int(w_w),
                    'height': int(h_w)
                },
                'contour': simplified_contour.squeeze().tolist()
            }
            
            maze_structure['walls'].append(wall_data)
            print(f"  Wall {i}: Area={area:.0f} pixels")
    
    # Add statistics
    maze_structure['statistics'] = {
        'total_compartments': len(maze_structure['compartments']),
        'total_walls': len(maze_structure['walls']),
        'total_navigable_area': sum(c['area'] for c in maze_structure['compartments']),
        'total_wall_area': sum(w['area'] for w in maze_structure['walls'])
    }
    
    print("\n" + "="*60)
    print("MAZE STRUCTURE SUMMARY:")
    print("="*60)
    print(f"Total compartments: {maze_structure['statistics']['total_compartments']}")
    print(f"Total walls: {maze_structure['statistics']['total_walls']}")
    print(f"Total navigable area: {maze_structure['statistics']['total_navigable_area']:.0f} pixels")
    print(f"Total wall area: {maze_structure['statistics']['total_wall_area']:.0f} pixels")
    
    # Visualize
    visualize_maze_structure(frame_roi, maze_structure)
    
    return maze_structure

def visualize_maze_structure(frame, maze_structure):
    """
    Visualize the extracted maze structure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original frame
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Frame', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Compartments
    ax2 = axes[1]
    compartment_img = np.zeros_like(frame)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(maze_structure['compartments'])))
    
    for i, comp in enumerate(maze_structure['compartments']):
        contour = np.array(comp['contour']).reshape(-1, 1, 2).astype(np.int32)
        color = (colors[i][:3] * 255).astype(np.uint8).tolist()
        cv2.drawContours(compartment_img, [contour], -1, color, -1)  # Fill
        cv2.drawContours(compartment_img, [contour], -1, (255, 255, 255), 2)  # Border
        
        # Draw centroid
        cx, cy = comp['centroid']['x'], comp['centroid']['y']
        cv2.circle(compartment_img, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(compartment_img, str(comp['id']), (cx-5, cy+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    ax2.imshow(cv2.cvtColor(compartment_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Compartments ({len(maze_structure["compartments"])})', 
                 fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Walls
    ax3 = axes[2]
    wall_img = np.zeros_like(frame)
    
    for wall in maze_structure['walls']:
        contour = np.array(wall['contour']).reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(wall_img, [contour], -1, (100, 100, 100), -1)  # Fill
        cv2.drawContours(wall_img, [contour], -1, (255, 0, 0), 2)  # Border
    
    ax3.imshow(cv2.cvtColor(wall_img, cv2.COLOR_BGR2RGB))
    ax3.set_title(f'Walls ({len(maze_structure["walls"])})', 
                 fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_maze_structure(maze_structure, output_path):
    """
    Save maze structure to JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(maze_structure, f, indent=2)
    
    print(f"\n✓ Maze structure saved to: {output_path}")
    
    # Print file size
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"  File size: {file_size:.2f} KB")

def load_maze_structure(json_path):
    """
    Load maze structure from JSON file
    """
    with open(json_path, 'r') as f:
        maze_structure = json.load(f)
    
    print(f"✓ Maze structure loaded from: {json_path}")
    print(f"  Compartments: {maze_structure['statistics']['total_compartments']}")
    print(f"  Walls: {maze_structure['statistics']['total_walls']}")
    
    return maze_structure

def check_mouse_compartment(mouse_position, maze_structure):
    """
    Determine which compartment the mouse is in
    
    Parameters:
    -----------
    mouse_position : tuple
        (x, y) coordinates of mouse
    maze_structure : dict
        Maze structure dictionary
    
    Returns:
    --------
    compartment_id : int or None
        ID of compartment mouse is in, or None if not in any
    """
    mx, my = mouse_position
    point = np.array([[mx, my]], dtype=np.int32)
    
    for comp in maze_structure['compartments']:
        contour = np.array(comp['contour']).reshape(-1, 1, 2).astype(np.int32)
        
        # Check if point is inside contour
        result = cv2.pointPolygonTest(contour, (mx, my), False)
        
        if result >= 0:  # Inside or on boundary
            return comp['id']
    
    return None

def analyze_trajectory_with_maze(trajectory, maze_structure):
    """
    Analyze trajectory in context of maze structure
    
    Parameters:
    -----------
    trajectory : np.array
        Array of (x, y) positions
    maze_structure : dict
        Maze structure dictionary
    
    Returns:
    --------
    analysis : dict
        Analysis results
    """
    compartment_visits = {comp['id']: 0 for comp in maze_structure['compartments']}
    compartment_time = {comp['id']: 0 for comp in maze_structure['compartments']}
    trajectory_compartments = []
    
    print("\nAnalyzing trajectory with maze structure...")
    
    for i, (x, y) in enumerate(trajectory):
        comp_id = check_mouse_compartment((x, y), maze_structure)
        trajectory_compartments.append(comp_id)
        
        if comp_id is not None:
            compartment_time[comp_id] += 1
            
            # Count visits (new visit if different from previous compartment)
            if i == 0 or (i > 0 and trajectory_compartments[i-1] != comp_id):
                compartment_visits[comp_id] += 1
    
    analysis = {
        'compartment_visits': compartment_visits,
        'compartment_time': compartment_time,
        'trajectory_compartments': trajectory_compartments,
        'total_frames': len(trajectory),
        'frames_in_compartments': sum(1 for c in trajectory_compartments if c is not None)
    }
    
    print("\nCompartment Analysis:")
    print("="*60)
    for comp_id in sorted(compartment_visits.keys()):
        visits = compartment_visits[comp_id]
        time = compartment_time[comp_id]
        if time > 0:
            print(f"Compartment {comp_id}: {visits} visits, {time} frames ({100*time/len(trajectory):.1f}%)")
    
    return analysis

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    video_path = "C:/Users/aleja/Box/Awake Project/Maze data/simplermaze/mouse 6357/2024-08-28_11_58_146357session3.6/segments/6357_2024-08-28_11_58_14s3.6_trial_008.mp4"
    roi_file = "C:/Users/aleja/Desktop/maze_roi.npy"
    output_folder = "C:/Users/aleja/Desktop/"
    
    # Load ROI
    if not os.path.exists(roi_file):
        print(f"ERROR: ROI file not found: {roi_file}")
        print("Please run the main script first to select and save the ROI")
        exit()
    
    roi = tuple(np.load(roi_file))
    print(f"Loaded ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
    
    # Extract maze structure
    print("\n" + "="*60)
    print("EXTRACTING MAZE STRUCTURE")
    print("="*60)
    
    maze_structure = extract_maze_structure(
        video_path, 
        roi, 
        frame_number=0,  # Use first frame
        maze_threshold=160  # Adjust if needed
    )
    
    if maze_structure:
        # Save to JSON
        json_path = os.path.join(output_folder, "maze_structure.json")
        save_maze_structure(maze_structure, json_path)
        
        # Example: Load it back
        print("\n" + "="*60)
        print("TESTING LOAD FUNCTIONALITY")
        print("="*60)
        loaded_structure = load_maze_structure(json_path)
        
        # Example: Analyze a trajectory (if you have one)
        trajectory_file = os.path.join(output_folder, "trial_008_trajectory.npy")
        if os.path.exists(trajectory_file):
            print("\n" + "="*60)
            print("ANALYZING TRAJECTORY WITH MAZE STRUCTURE")
            print("="*60)
            trajectory = np.load(trajectory_file)
            analysis = analyze_trajectory_with_maze(trajectory, loaded_structure)
            
            # Save analysis
            analysis_path = os.path.join(output_folder, "trajectory_maze_analysis.json")
            with open(analysis_path, 'w') as f:
                # Convert numpy types to native Python types for JSON
                json_analysis = {
                    'compartment_visits': {int(k): int(v) for k, v in analysis['compartment_visits'].items()},
                    'compartment_time': {int(k): int(v) for k, v in analysis['compartment_time'].items()},
                    'total_frames': int(analysis['total_frames']),
                    'frames_in_compartments': int(analysis['frames_in_compartments'])
                }
                json.dump(json_analysis, f, indent=2)
            print(f"\n✓ Trajectory analysis saved to: {analysis_path}")
        else:
            print(f"\nNo trajectory file found at {trajectory_file}")
            print("Run the main tracking script first to generate trajectory data")
    else:
        print("✗ Failed to extract maze structure")