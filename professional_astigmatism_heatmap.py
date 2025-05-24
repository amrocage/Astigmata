# professional_astigmatism_heatmap.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
from matplotlib.patches import Circle
import os

def generate_corneal_map(
    cylinder_power: float,
    axis_degrees: int,
    map_type: str, 
    output_filename: str = "corneal_map.png",
    patient_id: str = "N/A",
    eye_type: str = "OS",
    k_mean: float = 44.5, 
    bfs_front: float = 7.15, 
    bfs_back: float = 5.98,   
    central_thickness_sim: float = 540, 
    max_radius_mm: float = 4.5, 
    img_size_px: int = 400, 
    dpi: int = 100,          
    astigmatism_viz_threshold: float = 0.01 
):
    """
    Generates an further enhanced simulated corneal map, aiming for a professional look
    similar to the reference image.
    """
    abs_cylinder = abs(cylinder_power)
    is_significant_astigmatism = abs_cylinder >= astigmatism_viz_threshold 

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if cylinder_power < 0: 
        steeper_meridian_deg = (axis_degrees + 90) % 180
        k_steep = k_mean + abs_cylinder / 2.0
        k_flat = k_mean - abs_cylinder / 2.0
    else: 
        steeper_meridian_deg = axis_degrees
        k_steep = k_mean + abs_cylinder / 2.0
        k_flat = k_mean - abs_cylinder / 2.0
    
    steeper_meridian_rad = np.deg2rad(steeper_meridian_deg)

    num_points = 256  
    r_coords = np.linspace(0, max_radius_mm, num_points)
    theta_coords = np.linspace(0, 2 * np.pi, num_points)
    R, Theta = np.meshgrid(r_coords, theta_coords)
    X = R * np.cos(Theta) 
    Y = R * np.sin(Theta) 

    data_values = np.zeros_like(R)
    cmap_to_use = None
    norm_to_use = None
    colorbar_ticks = None
    colorbar_label = ""
    plot_map_title = ""
    unit_label = "" 
    cbar_tick_labels = None
    cbar_is_discrete = False

    # --- Map Type Specific Data Simulation, Colormap, and Normalization ---
    if map_type == "axial_curvature":
        plot_map_title = "Axial / Sagittal Curvature (Front)"
        data_values = k_mean + (abs_cylinder / 2.0) * np.cos(2 * (Theta - steeper_meridian_rad))
        
        colors_axial = [ 
            (0.00, "#000080"), (0.05, "#0000C0"), (0.10, "#0000FF"), (0.15, "#0055FF"), 
            (0.20, "#00AAFF"), (0.25, "#00FFFF"), (0.30, "#55FFAA"), (0.35, "#AAFF55"), 
            (0.40, "#00FF00"), (0.45, "#55FF00"), (0.50, "#AAFF00"), (0.55, "#FFFF00"), 
            (0.60, "#FFDD00"), (0.65, "#FFBF00"), (0.70, "#FFAA00"), (0.75, "#FF7F00"), 
            (0.80, "#FF5500"), (0.85, "#FF0000"), (0.90, "#D40000"), (0.95, "#AA0000"), 
            (1.00, "#800000")
        ]
        cmap_to_use = LinearSegmentedColormap.from_list("axial_cmap_pro_v3", colors_axial)
        
        vmin_val, vmax_val = 38.0, 52.0 
        norm_to_use = plt.Normalize(vmin=vmin_val, vmax=vmax_val)
        
        k_min_sim_on_map = np.min(data_values[R <= max_radius_mm * 0.8]) # Min K within central 80%
        k_values_text = (f"Kmean: {k_mean:.2f}\n"
                         f"Ksteep: {k_steep:.2f} D @ {steeper_meridian_deg}°\n"
                         f"Kflat:  {k_flat:.2f} D @ {(steeper_meridian_deg-90+180)%180}°\n" 
                         f"Min K: {k_min_sim_on_map:.2f} D")

        colorbar_label = "Corneal Power (D)"
        unit_label = " D" 
        specific_ticks_axial = np.array([38, 40, 41.5, 43, 44.5, 46, 47.5, 49, 50.5, 52])
        colorbar_ticks = [t for t in specific_ticks_axial if vmin_val <= t <= vmax_val]
        if not colorbar_ticks or len(colorbar_ticks) < 3:
             colorbar_ticks = MaxNLocator(nbins=9).tick_values(vmin_val, vmax_val)

    elif map_type == "elevation_front" or map_type == "elevation_back":
        is_front = map_type == "elevation_front"
        plot_map_title = f"Elevation ({'Front' if is_front else 'Back'})"
        bfs_val = bfs_front if is_front else bfs_back
        plot_map_title += f"\nBFS={bfs_val:.2f} Float, Dia=8.00mm"

        elevation_scale = 30 * abs_cylinder 
        x_rot = X * np.cos(steeper_meridian_rad) - Y * np.sin(steeper_meridian_rad)
        y_rot = X * np.sin(steeper_meridian_rad) + Y * np.cos(steeper_meridian_rad)
        astig_component = (x_rot**2 - y_rot**2) / (max_radius_mm**1.8) 
        spherical_component = -20 * (R / max_radius_mm)**2.2 
        if not is_front: spherical_component *= -0.7 
        data_values = elevation_scale * astig_component * 0.1 + spherical_component
        if is_significant_astigmatism and abs_cylinder > 0.75:
            coma_like = 8 * (R/max_radius_mm)**2.8 * np.cos(Theta - np.deg2rad(axis_degrees+30)) * (abs_cylinder/1.2)
            data_values += coma_like
            if not is_front: data_values += np.random.normal(0, 2, data_values.shape) * (R/max_radius_mm)
        data_values -= np.mean(data_values)
        data_values = np.clip(data_values, -85, 85)

        colors_elev = [ 
            "#000080", "#0000C0", "#0000FF", "#0055FF", "#00AAFF", 
            "#00FFFF", "#55FFAA", "#AAFF55", 
            "#FFFFFF", 
            "#FFDD55", "#FFAA55", "#FF5500", 
            "#FF0000", "#AA0000", "#800000"  
        ]
        boundaries = np.array([-75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75])
        cmap_to_use = ListedColormap(colors_elev)
        norm_to_use = BoundaryNorm(boundaries, ncolors=cmap_to_use.N, clip=True)
        colorbar_ticks = boundaries 
        cbar_tick_labels = [str(int(b)) for b in boundaries]
        colorbar_label = "Elevation" 
        unit_label = "µm"
        cbar_is_discrete = True


    elif map_type == "corneal_thickness":
        plot_map_title = "Corneal Thickness"
        min_thick_sim = central_thickness_sim - 25 * (abs_cylinder / 2.5) 
        max_thick_sim = central_thickness_sim + 90
        offset_r = 0.7 
        offset_angle = steeper_meridian_rad + np.pi/2.2 
        x_center_offset = offset_r * np.cos(offset_angle)
        y_center_offset = offset_r * np.sin(offset_angle)
        R_from_thinnest = np.sqrt((X - x_center_offset)**2 + (Y - y_center_offset)**2)
        data_values = min_thick_sim + (max_thick_sim - min_thick_sim) * (R_from_thinnest / (max_radius_mm + offset_r))**1.1
        if is_significant_astigmatism:
            data_values += 15 * (abs_cylinder/2.0) * np.cos(2 * (Theta - steeper_meridian_rad - np.pi/4)) * (R/max_radius_mm)
        data_values += np.random.normal(0, 3, data_values.shape) 
        data_values = np.clip(data_values, 400, 750) 

        colors_thick = ["#007FFF","#00BFFF", "#00FFFF", "#7FFFD4", "#00FF7F", "#7FFF00", "#FFFF00", "#FFD700", "#FFA500", "#FF7F50", "#FF4500", "#FF0000", "#B22222"]
        cmap_to_use = LinearSegmentedColormap.from_list("thickness_cmap_pro_v3", colors_thick)
        vmin_val, vmax_val = 430, 670
        norm_to_use = plt.Normalize(vmin=vmin_val, vmax=vmax_val)
        colorbar_label = "Thickness"
        unit_label = "µm"
        colorbar_ticks = MaxNLocator(nbins=8).tick_values(vmin_val, vmax_val)
    else:
        print(f"Unknown map type: {map_type}"); return None

    fig = plt.figure(figsize=(img_size_px / dpi, (img_size_px + 70) / dpi), dpi=dpi) 
    ax = fig.add_subplot(111, projection='polar')
    fig.patch.set_facecolor('#2a2a2a') 
    
    im = ax.pcolormesh(Theta, R, data_values, cmap=cmap_to_use, norm=norm_to_use, shading='gouraud', zorder=0)

    ax.set_rmax(max_radius_mm)
    rticks_mm_values = np.array([1.5, 3.0, 4.0]) 
    ax.set_rticks(rticks_mm_values) 
    ax.set_rlabel_position(-25) 
    ax.set_yticklabels([f'{int(t*2)}mm' for t in rticks_mm_values], fontsize=7, color='lightgrey')
    
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f'{deg}°' for deg in np.arange(0, 360, 30)], fontsize=7, color='lightgrey')
    ax.grid(True, linestyle=':', color='#666666', alpha=0.8, zorder=1, linewidth=0.5)
    ax.spines['polar'].set_edgecolor('#777777')
    ax.set_facecolor('#303030') 

    label_dist_factor = 1.45 
    font_props_eye_labels = {'fontsize': 9, 'fontweight': 'bold', 'color': '#E0E0E0'}
    if eye_type.upper() == "OS": 
        ax.text(np.deg2rad(0), max_radius_mm * label_dist_factor, "N", ha='center', va='center', **font_props_eye_labels)
        ax.text(np.deg2rad(180), max_radius_mm * label_dist_factor, "T", ha='center', va='center', **font_props_eye_labels)
    elif eye_type.upper() == "OD": 
        ax.text(np.deg2rad(0), max_radius_mm * label_dist_factor, "T", ha='center', va='center', **font_props_eye_labels)
        ax.text(np.deg2rad(180), max_radius_mm * label_dist_factor, "N", ha='center', va='center', **font_props_eye_labels)
    
    ax.text(np.deg2rad(45), max_radius_mm * 0.82, eye_type.upper(), 
            ha='center', va='center', fontsize=16, fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.3", fc="#303030", ec="#666666", lw=1, alpha=0.75))

    fig.text(0.5, 0.97, plot_map_title, ha='center', va='top', fontsize=11, color='white', fontweight='bold')

    if map_type == "axial_curvature" and 'k_values_text' in locals():
        fig.text(0.03, 0.03, k_values_text, fontsize=7, color='#FFFFB0', linespacing=1.4,
                 verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='black', alpha=0.7))

    # --- Colorbar Customization ---
    cbar_width = 0.035
    cbar_ax = fig.add_axes([0.90, 0.12, cbar_width, 0.75]) 
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=colorbar_ticks)
    cbar.ax.tick_params(labelsize=7, colors='white')
    
    if cbar_is_discrete and cbar_tick_labels:
        cbar.ax.set_yticklabels(cbar_tick_labels)
        # Add text labels next to discrete color blocks (approximating Pentacam style)
        tick_locs = cbar.get_ticks()
        for i, loc in enumerate(tick_locs):
            if i < len(cbar_tick_labels): # Ensure we don't go out of bounds
                # Position text to the right of the colorbar
                cbar.ax.text(1.6, loc, cbar_tick_labels[i], ha='left', va='center', color='white', fontsize=6.5)
        # Remove the default ticks and labels if we added custom text labels
        cbar.set_ticks([]) 
        # cbar.set_label("", labelpad=-10) # Remove default label if using custom text

    if unit_label:
         cbar.ax.text(0.5, 1.08, unit_label, transform=cbar.ax.transAxes, 
                      fontsize=8, color='white', ha='center', va='bottom', fontweight='bold')
    
    if map_type.startswith("elevation"):
        cbar.ax.text(0.5, -0.12, "Elevation\nHeight", transform=cbar.ax.transAxes,
                     fontsize=7, color='white', ha='center', va='top', linespacing=1.3)
        if boundaries is not None and len(boundaries) > 1:
            step_val = abs(boundaries[1]-boundaries[0])
            cbar.ax.text(0.5, -0.18, f"{step_val} {unit_label} / step", transform=cbar.ax.transAxes,
                         fontsize=7, color='white', ha='center', va='top')
    
    plt.subplots_adjust(left=0.02, right=0.82, top=0.88, bottom=0.02) # Adjusted right for wider colorbar area

    try:
        plt.savefig(output_filename, dpi=dpi, facecolor=fig.get_facecolor())
        print(f"Successfully saved {map_type} map to {output_filename}")
        plt.close(fig)
        return os.path.abspath(output_filename)
    except Exception as e:
        print(f"Error saving {map_type} map to {output_filename}: {e}")
        plt.close(fig)
        return None

if __name__ == "__main__":
    print("Generating example enhanced corneal maps (v6)...")
    output_dir_examples = "outputs/corneal_maps_v6_pro_examples" 
    os.makedirs(output_dir_examples, exist_ok=True)
    patient_data_examples = [
        {"id": "P001_HighAstig", "cyl": -4.75, "axis": 172, "eye": "OS", "k_mean": 41.5, "bfs_f": 6.95, "bfs_b": 5.70, "cct": 505},
        {"id": "P002_ModAstig", "cyl": 2.25, "axis": 88, "eye": "OD", "k_mean": 46.1, "bfs_f": 7.35, "bfs_b": 6.15, "cct": 565},
        {"id": "P003_LowAstig", "cyl": -0.60, "axis": 25, "eye": "OS", "k_mean": 43.8, "bfs_f": 7.10, "bfs_b": 5.90, "cct": 530},
        {"id": "P004_NoAstig", "cyl": 0.10, "axis": 95, "eye": "OD", "k_mean": 44.0, "bfs_f": 7.18, "bfs_b": 5.95, "cct": 550}
    ]
    map_types_to_generate = ["axial_curvature", "elevation_front", "corneal_thickness", "elevation_back"]
    for p_data in patient_data_examples:
        for mt in map_types_to_generate:
            fname = os.path.join(output_dir_examples, f"{p_data['id']}_{p_data['eye']}_{mt}.png")
            generate_corneal_map(
                cylinder_power=p_data["cyl"], axis_degrees=p_data["axis"], map_type=mt,
                output_filename=fname, patient_id=p_data["id"], eye_type=p_data["eye"],
                k_mean=p_data["k_mean"], bfs_front=p_data["bfs_f"], bfs_back=p_data["bfs_b"],
                central_thickness_sim=p_data["cct"], img_size_px=400, dpi=100 
            )
    print(f"Example maps generated in '{os.path.abspath(output_dir_examples)}'.")

