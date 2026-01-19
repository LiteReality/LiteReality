import os
from PIL import Image

def merge_image(seg_folder, org_folder, legend_path, output_folder):
    # Load the legend image
    legend_img = Image.open(legend_path)
    # double size of legend image
    legend_img = legend_img.resize((legend_img.width * 2, legend_img.height * 2))
    # Iterate through images in rendering_org
    for filename in os.listdir(org_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the corresponding images
            org_img = Image.open(os.path.join(org_folder, filename))
            seg_img = Image.open(os.path.join(seg_folder, filename))
            
            # Ensure the images have the same height
            width, height = org_img.size
            seg_img = seg_img.resize((width, height))

            # Create a new blank image with enough width for both org and seg side by side
            new_img_width = width * 2
            new_img = Image.new("RGB", (new_img_width, height))

            # Paste images side by side
            new_img.paste(org_img, (0, 0))
            new_img.paste(seg_img, (width, 0))

            # Calculate position for the legend at the bottom right corner
            legend_x = int(width  - legend_img.width * 0.5 ) # Small offset from the right edge
            legend_y = int(height /2  - legend_img.height * 0.5 )   # Small offset from the bottom edge

            # Ensure transparency is handled correctly
            new_img.paste(legend_img, (legend_x, legend_y), legend_img.convert("RGBA"))
            # Save the combined image
            new_img.save(os.path.join(output_folder, filename))

    print("Images combined successfully with the legend overlayed at the bottom left!")


def center_obj(obj_file_path, center):
    """Centers the .obj file by translating all vertices to make the object centered at the origin."""
    with open(obj_file_path, 'r') as file:
        lines = file.readlines()

    # Accumulate vertices
    vertices = []
    for line in lines:
        if line.startswith('v '):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))

    centroid_x, centroid_y, centroid_z = center

    # Calculate centroid
    if vertices:
        # centroid_x = sum(v[0] for v in vertices) / len(vertices)
        # centroid_y = sum(v[1] for v in vertices) / len(vertices)
        # centroid_z = sum(v[2] for v in vertices) / len(vertices)

        # Translate vertices to center them
        centered_lines = []
        for line in lines:
            if line.startswith('v '):
                _, x, y, z = line.split()
                new_x = float(x) - centroid_x
                new_y = float(y) - centroid_y
                new_z = float(z) - centroid_z
                centered_lines.append(f"v {new_x} {new_y} {new_z}\n")
            else:
                centered_lines.append(line)

        # Save centered content to a temporary file
        centered_obj_file_path = obj_file_path.replace('.obj', '_centered.obj')
        with open(centered_obj_file_path, 'w') as centered_file:
            centered_file.writelines(centered_lines)

        return centered_obj_file_path
    return obj_file_path


def separate_obj_by_group(input_filepath, output_folder):
    """
    Separates an OBJ file into parts based on group (g) markers if they exist.
    If no group markers are found, it uses usemtl markers instead.
    Each output file is saved into the output_folder with the file name equal
    to the group/material name and a .obj extension.
    Each file will include the global definitions (vertices, texture coordinates,
    normals, material libraries, object names, and comments) to ensure the separated
    file can be loaded without missing data.
    
    :param input_filepath: Path to the input OBJ file.
    :param output_folder: Path to the output folder where separated OBJ files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the entire file
    with open(input_filepath, 'r') as file:
        lines = file.readlines()
    
    # Define which lines are considered global
    def is_global(line):
        return (line.startswith("v ") or
                line.startswith("vt ") or
                line.startswith("vn ") or
                line.startswith("mtllib") or
                line.startswith("o ") or
                line.startswith("#"))
    
    # First, collect all global definitions
    global_lines = [line for line in lines if is_global(line)]
    
    # Decide which marker to use for separating sections:
    grouping_mode = any(line.startswith("g ") for line in lines)
    marker = "g " if grouping_mode else "usemtl"
    
    sections = {}
    current_key = None
    
    # Process the file lines (we include only non-global lines here)
    for line in lines:
        # Skip global lines so they are not duplicated in sections.
        if is_global(line):
            continue
        
        if grouping_mode:
            if line.startswith("g "):
                # Use group name as key (e.g., "g MyGroup" becomes "MyGroup")
                parts = line.strip().split(maxsplit=1)
                current_key = parts[1] if len(parts) > 1 else "default_group"
                # Initialize section with the group marker.
                sections.setdefault(current_key, []).append(line)
            elif line.startswith("usemtl"):
                # Optionally, you might want to record usemtl lines in the current section.
                # If there is no current section, use a default key.
                if current_key is None:
                    current_key = "default_group"
                    sections.setdefault(current_key, [])
                sections[current_key].append(line)
            elif line.startswith("f ") or line.startswith("s ") or line.startswith("l "):
                # Face, smoothing, or line definitions – add to current group.
                if current_key is None:
                    current_key = "default_group"
                    sections.setdefault(current_key, [])
                sections[current_key].append(line)
            else:
                # Other lines can be appended to current section if desired.
                if current_key is not None:
                    sections[current_key].append(line)
        else:
            # Fallback: use material markers ("usemtl") for sectioning.
            if line.startswith("usemtl"):
                parts = line.strip().split(maxsplit=1)
                current_key = parts[1] if len(parts) > 1 else "default_material"
                sections.setdefault(current_key, []).append(line)
            elif line.startswith("f ") or line.startswith("s ") or line.startswith("l "):
                if current_key is None:
                    current_key = "default_material"
                    sections.setdefault(current_key, [])
                sections[current_key].append(line)
            else:
                # Add any other non-global line to the current section if available.
                if current_key is not None:
                    sections[current_key].append(line)
    
    # Write out each section, combining the global definitions with the section-specific lines.
    for key, section_lines in sections.items():
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        out_filename = os.path.join(output_folder, f"{safe_key}.obj")
        with open(out_filename, 'w') as outfile:
            outfile.writelines(global_lines)
            outfile.write("\n")  # Separate global section from per-section data
            outfile.writelines(section_lines)
        print(f"Wrote file: {out_filename}")


