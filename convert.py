import os


def convert_tsp_format(folder_path, output_folder=None):
    # Create output folder if specified and doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".tsp"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            in_node_section = False
            node_coords = []

            for line in lines:
                clean_line = line.strip()

                # 1. Detect start of Node Coord Section
                if clean_line.startswith("NODE_COORD_SECTION"):
                    in_node_section = True
                    # Change the header to your custom field name
                    new_lines.append("NODE_LOCATION:\n")
                    continue

                # 2. Detect end of Node Coord Section (start of weights or other fields)
                if in_node_section and (
                    clean_line.startswith("EDGE_WEIGHT_SECTION")
                    or ":" in clean_line
                    or clean_line == "EOF"
                ):
                    in_node_section = False
                    # Add the cleaned coordinates (one pair per line, no index)
                    new_lines.extend(node_coords)

                if in_node_section:
                    # Extract coords: "1 47.419038 8.538454" -> "47.419038 8.538454"
                    parts = clean_line.split()
                    if len(parts) >= 3:
                        lat, lon = parts[1], parts[2]
                        node_coords.append(f"{lat} {lon}\n")
                else:
                    # Keep all other lines as they are
                    new_lines.append(line)

            # Determine where to save
            save_path = (
                os.path.join(output_folder, filename) if output_folder else file_path
            )

            with open(save_path, "w") as f:
                f.writelines(new_lines)

            print(f"Converted: {filename}")


# Usage
# Set output_folder to a new path if you want to keep backups!
#
#
for n in [10, 100]:  # [10,100, 1000, 10000, 200, 2000, 25, 50, 500, 5000]:
    convert_tsp_format(f"/home/schafhdaniel@edu.local/tsp-solvers/data/tsp_dataset/{n}")
