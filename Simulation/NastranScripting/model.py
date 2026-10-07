from pathlib import Path

from pyNastran.bdf.bdf import BDF


def load_model(input_file: Path) -> BDF:
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Basismodell nicht gefunden:\n{input_file}"
        )

    model = BDF(debug=False)
    model.read_bdf(
        str(input_file),
        xref=False,
        validate=True,
    )

    return model


def change_forced_displacement_by_id(
    model: BDF,
    forced_displacement_id: int,
    new_value: float,
) -> None:
    """
    Changes the forced displacement identified by its SPC/SPCD SID.

    This assumes the SID belongs exclusively to one forced displacement.
    """

    changed_entries = 0

    # SPCD entries are stored in model.loads.
    for card in model.loads.get(forced_displacement_id, []):
        if card.type != "SPCD":
            continue

        for index, old_value in enumerate(card.enforced):
            card.enforced[index] = new_value
            changed_entries += 1

            print(
                f"SPCD changed: SID={forced_displacement_id}, "
                f"GRID={card.node_ids[index]}, "
                f"DOF={card.components[index]}, "
                f"{float(old_value):g} -> {new_value:g}"
            )

    # Non-zero SPC entries are stored in model.spcs.
    for card in model.spcs.get(forced_displacement_id, []):
        if card.type != "SPC":
            continue

        for index, old_value in enumerate(card.enforced):
            card.enforced[index] = new_value
            changed_entries += 1

            print(
                f"SPC changed: SID={forced_displacement_id}, "
                f"GRID={card.node_ids[index]}, "
                f"DOF={card.components[index]}, "
                f"{float(old_value):g} -> {new_value:g}"
            )

    if changed_entries == 0:
        raise RuntimeError(
            "No forced displacement found for "
            f"SID={forced_displacement_id}."
        )

    if changed_entries > 1:
        print(
            f"Warning: SID={forced_displacement_id} contains "
            f"{changed_entries} entries. All were changed."
        )


def change_forced_displacement_by_position(
    model: BDF,
    position_x: float,
    position_z: float,
    new_value: float,
    tolerance: float = 1e-3,
) -> None:
    """Change the forced displacement attached to a GRID position.

    The x and z coordinates must use the Nastran model coordinate system. A
    numeric tolerance is used because Nastran rewrites coordinates with
    floating-point formatting.
    Exactly one SPC/SPCD card must reference a matching GRID.
    """

    matching_cards = []
    for cards in (model.loads.values(), model.spcs.values()):
        for card_group in cards:
            for card in card_group:
                if card.type not in {"SPCD", "SPC"}:
                    continue

                matching_indices = []
                for index, node_id in enumerate(card.node_ids):
                    node = model.nodes.get(node_id)
                    if node is None:
                        continue

                    coordinates = node.get_position()
                    if (
                        abs(float(coordinates[0]) - position_x) <= tolerance
                        and abs(float(coordinates[2]) - position_z) <= tolerance
                    ):
                        matching_indices.append(index)

                if matching_indices:
                    matching_cards.append((card, matching_indices))

    if not matching_cards:
        raise RuntimeError(
            "No SPC/SPCD forced displacement found at "
            f"x={position_x}, z={position_z} "
            f"within tolerance {tolerance}."
        )

    if len(matching_cards) > 1:
        raise RuntimeError(
            "Multiple SPC/SPCD forced displacements found at "
            f"x={position_x}, z={position_z} "
            f"within tolerance {tolerance}."
        )

    card, matching_indices = matching_cards[0]
    for index in matching_indices:
        old_value = card.enforced[index]
        card.enforced[index] = new_value
        # print(
        #     f"{card.type} changed: GRID={card.node_ids[index]}, "
        #     f"DOF={card.components[index]}, "
        #     f"{float(old_value):g} -> {new_value:g}"
        # )


def save_model(model: BDF, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model.write_bdf(
        out_filename=str(output_file),
        size=16,
        is_double=False,
        interspersed=True,
        enddata=True,
    )

    print(f"Modell gespeichert: {output_file}")
