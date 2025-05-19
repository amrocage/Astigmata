# heatmap.py
import matplotlib.pyplot as plt
import numpy as np
import os
from PyQt5.QtCore import QSize # Import QSize for potential use in scaling logic if needed later

def generate_astigmatism_heatmap_viz(cylinder, axis, patient_id, eye, output_dir="temp_heatmaps"):
    """
    Generates a simplified visual representation of astigmatism as an image file
    based on Cylinder and Axis values.

    Args:
        cylinder (float): The cylinder value (magnitude of astigmatism).
        axis (int): The axis value in degrees (0-180).
        patient_id (str): The patient ID (for filename).
        eye (str): The eye (OD/OS) (for filename).
        output_dir (str): Directory to save the generated images.

    Returns:
        str: The absolute path to the generated image file, or None if no significant astigmatism.
    """
    # Define a threshold for visualization - only visualize significant astigmatism
    # This should match your ASTIGMATISM_THRESHOLD used in processing in Source.py
    ASTIGMATISM_VIZ_THRESHOLD = 0.25

    # Only generate heatmap if astigmatism is detected above the threshold
    if abs(cylinder) < ASTIGMATISM_VIZ_THRESHOLD:
        print(f"Heatmap Viz: Skipping generation for {patient_id} {eye} (Cylinder {cylinder:.2f} below threshold {ASTIGMATISM_VIZ_THRESHOLD}).")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique and safe filename for the image
    # Replace characters that might be problematic in filenames
    safe_patient_id = "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in str(patient_id))
    safe_eye = "".join(c if c.isalnum() else "_" for c in str(eye))
    # Use a consistent format for the filename
    output_filename = f"viz_{safe_patient_id}_{safe_eye}_cyl{abs(cylinder):.2f}_axis{axis}.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Heatmap Viz: Generating heatmap for {patient_id} {eye} (Cylinder: {cylinder:.2f}, Axis: {axis}). Saving to {output_path}")

    # --- Matplotlib Visualization Logic ---
    # Create a figure and axes without displaying them
    # Adjust figsize and dpi for the desired output image size and resolution
    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100) # Smaller figure size, higher dpi

    # Set background to transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Turn off axes
    ax.set_axis_off()

    # Set limits to keep the visualization centered
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal

    # Normalize cylinder value for color/intensity (example scaling)
    # Assuming a maximum cylinder value for scaling the visualization intensity.
    # You might need to adjust max_cylinder based on the typical range of your data.
    max_cylinder = 4.0 # Example max cylinder value (adjust as needed)
    # Scale intensity between a minimum value (e.g., 0.3) and 1.0
    # This ensures even low astigmatism above threshold has some visibility
    intensity = max(0.3, min(abs(cylinder) / max_cylinder, 1.0))

    # Choose a color based on intensity - using a simple gradient from a base color
    # Example: Gradient from light red to dark red
    base_color = np.array([1.0, 0.2, 0.2]) # Red
    # Interpolate towards white (or another color) for lower intensity
    final_color = base_color * intensity + (1 - intensity) * np.array([1.0, 0.8, 0.8]) # Mix with light red/pink
    color = (final_color[0], final_color[1], final_color[2], 0.9) # Add alpha

    # Draw a simplified 'bow-tie' shape often associated with astigmatism topography
    # This is a conceptual representation, not a precise optical simulation

    # Define points for a basic bow-tie shape
    # These points are relative to the center (0,0)
    bow_tie_points = np.array([
        [0, 0.5], [0.2, 0.3], [0.5, 0], [0.2, -0.3], [0, -0.5],
        [-0.2, -0.3], [-0.5, 0], [-0.2, 0.3], [0, 0.5] # Close the shape
    ]) * (0.8 * intensity + 0.2) # Scale the size based on intensity

    # Apply rotation based on the axis angle
    theta_rad = np.deg2rad(axis) # Convert axis angle to radians
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Rotate the points
    rotated_points = (R @ bow_tie_points.T).T # Apply rotation matrix

    # Add the rotated shape to the plot
    ax.fill(rotated_points[:, 0], rotated_points[:, 1], color=color, alpha=0.8, zorder=2)

    # Optionally draw a central point or circle
    ax.plot(0, 0, 'o', color='gray', markersize=4, zorder=3) # Central point

    # Save the figure to the specified path
    try:
        # Use bbox_inches='tight' and pad_inches=0 to remove extra whitespace
        # transparent=True makes the background transparent
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        print(f"Heatmap Viz: Successfully saved heatmap to {output_path}")
        plt.close(fig) # Close the figure to free up memory
        return os.path.abspath(output_path) # Return absolute path
    except Exception as e:
        print(f"Heatmap Viz: Error saving heatmap image to {output_path}: {e}")
        plt.close(fig) # Close the figure on error
        return None

# Example of how to use the function (for testing heatmap.py directly)
if __name__ == "__main__":
    # Generate a sample heatmap
    sample_cylinder = 1.5
    sample_axis = 45
    sample_patient_id = "P101"
    sample_eye = "OD"
    generated_path = generate_astigmatism_heatmap_viz(sample_cylinder, sample_axis, sample_patient_id, sample_eye)
    if generated_path:
        print(f"Generated sample heatmap at: {generated_path}")

    sample_cylinder_low = 0.1
    sample_axis_low = 90
    generated_path_low = generate_astigmatism_heatmap_viz(sample_cylinder_low, sample_axis_low, "P102", "OS")
    if generated_path_low is None:
        print(f"No heatmap generated for low cylinder: {sample_cylinder_low}")
