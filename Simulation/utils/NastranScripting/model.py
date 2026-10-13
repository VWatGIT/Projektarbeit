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
    """Change the translational forced displacement at a GRID position."""

    change_forced_value_by_position(
        model=model,
        position_x=position_x,
        position_z=position_z,
        component=2,
        new_value=new_value,
        tolerance=tolerance,
    )


def change_forced_value_by_position(
    model: BDF,
    position_x: float,
    position_z: float,
    component: int,
    new_value: float,
    tolerance: float = 1e-3,
) -> None:
    """Change one SPC/SPCD component at a GRID position.

    The x and z coordinates must use the Nastran model coordinate system. A
    numeric tolerance is used because Nastran rewrites coordinates with
    floating-point formatting. Components 1-3 are translations and
    components 4-6 are rotations.
    """

    if component not in range(1, 7):
        raise ValueError(f"Nastran component must be between 1 and 6: {component}")

    matching_entries = []
    matching_workpoint_cards = []
    for cards in (model.loads.values(), model.spcs.values()):
        for card_group in cards:
            for card in card_group:
                if card.type not in {"SPCD", "SPC"}:
                    continue

                workpoint_node_ids = set()
                for index, node_id in enumerate(card.node_ids):
                    node = model.nodes.get(node_id)
                    if node is None:
                        continue

                    coordinates = node.get_position()
                    if (
                        abs(float(coordinates[0]) - position_x) <= tolerance
                        and abs(float(coordinates[2]) - position_z) <= tolerance
                    ):
                        workpoint_node_ids.add(node_id)
                        if int(card.components[index]) == component:
                            matching_entries.append((card, index))

                if workpoint_node_ids:
                    matching_workpoint_cards.append((card, workpoint_node_ids))

    if len(matching_entries) > 1:
        raise RuntimeError(
            "Multiple SPC/SPCD components found at "
            f"x={position_x}, z={position_z}, component={component} "
            f"within tolerance {tolerance}: {len(matching_entries)} entries."
        )

    if not matching_entries:
        if len(matching_workpoint_cards) != 1:
            raise RuntimeError(
                "No unique SPCD workpoint found at "
                f"x={position_x}, z={position_z} for component={component} "
                f"within tolerance {tolerance}."
            )

        card, node_ids = matching_workpoint_cards[0]
        if card.type != "SPCD" or len(node_ids) != 1:
            raise RuntimeError(
                "Cannot add a forced component at "
                f"x={position_x}, z={position_z}: the matching workpoint "
                "is not one SPCD node."
            )

        node_id = next(iter(node_ids))
        card.node_ids.append(node_id)
        card.components.append(component)
        card.enforced.append(new_value)
        return

    card, index = matching_entries[0]
    card.enforced[index] = new_value


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
