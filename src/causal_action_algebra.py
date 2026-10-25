"""Public, finite semantics of interventions on two fixed proposed constants."""

ACTION_CODES = (
    "enable_first", "enable_second", "restore_first", "restore_second",
    "enable_both", "restore_both", "toggle_first", "toggle_second", "swap",
    "enable_first_if_second", "enable_second_if_first", "restore_first_if_second",
    "restore_second_if_first", "toggle_first_if_second", "toggle_second_if_first", "no_change",
)

# Bit 1 denotes the first proposed clamp, bit 2 the second. Values are unchanged.
TRANSITIONS = (
    (1, 1, 3, 3), (2, 3, 2, 3), (0, 0, 2, 2), (0, 1, 0, 1),
    (3, 3, 3, 3), (0, 0, 0, 0), (1, 0, 3, 2), (2, 3, 0, 1), (0, 2, 1, 3),
    (0, 1, 3, 3), (0, 3, 2, 3), (0, 1, 2, 2), (0, 1, 2, 1),
    (0, 1, 3, 2), (0, 3, 2, 1), (0, 1, 2, 3),
)

PROGRAM_SEMANTICS = (
    "Begin with neither proposed override active. Execute the numbered instructions in order. "
    "An active override replaces its equation by its fixed proposed constant, whether 0 or 1. "
    "Restoring a variable removes its override. Toggling a clamp changes whether it is active; "
    "it never flips its proposed bit. Conditions inspect clamp status immediately before that step. "
    "If a condition is false, change nothing. Swapping statuses exchanges which clamps are active, "
    "not their proposed values. At readout recompute the SCM with only the final active overrides, "
    "retaining the original individual's noise. Earlier overrides leave no other memory."
)

ACTION_SEMANTICS = (
    "Interpret ONE instruction as a function of the current clamp statuses. "
    "The current state is NOT supplied and may be any of the four possibilities; "
    "do not assume either clamp is initially active or inactive. "
    "An active override replaces its equation by its fixed proposed constant, whether 0 or 1. "
    "Restoring removes a clamp. Toggling reverses clamp presence, never its proposed bit. "
    "A condition checks the status before this instruction; if false, change nothing. "
    "Swapping statuses exchanges presence, not proposed values."
)


def fold_actions(actions, initial=0):
    if type(initial) is not int or initial not in range(4):
        raise ValueError("invalid initial intervention state")
    states = [initial]
    for action in actions:
        if type(action) is not int or action not in range(len(ACTION_CODES)):
            raise ValueError("invalid intervention transition")
        states.append(TRANSITIONS[action][states[-1]])
    return states


def intervention_mask(intervene, proposed):
    if set(intervene) - set(proposed):
        raise ValueError("only the two publicly proposed clamps may be active")
    return sum(1 << i for i, name in enumerate(proposed) if name in intervene)


def action_descriptions(first, second):
    f, s = first, second
    return [
        f"Activate {f}'s proposed clamp", f"Activate {s}'s proposed clamp",
        f"Restore {f}'s original equation", f"Restore {s}'s original equation",
        "Activate both proposed clamps", "Restore both original equations",
        f"Toggle whether {f}'s clamp is active", f"Toggle whether {s}'s clamp is active",
        f"Swap the current clamp statuses of {f} and {s}",
        f"If {s}'s clamp is active, activate {f}'s clamp; otherwise change nothing",
        f"If {f}'s clamp is active, activate {s}'s clamp; otherwise change nothing",
        f"If {s}'s clamp is active, restore {f}'s equation; otherwise change nothing",
        f"If {f}'s clamp is active, restore {s}'s equation; otherwise change nothing",
        f"If {s}'s clamp is active, toggle {f}'s clamp status; otherwise change nothing",
        f"If {f}'s clamp is active, toggle {s}'s clamp status; otherwise change nothing",
        "Leave both clamp statuses unchanged",
    ]


def render_action(action, first, second, language=False):
    if not language:
        plain = [
            f"Impose the proposed override for {first}.", f"Impose the proposed override for {second}.",
            f"Remove the override for {first}.", f"Remove the override for {second}.",
            "Impose both proposed overrides.", "Remove both overrides.",
            f"Switch the override for {first} from active to inactive, or from inactive to active.",
            f"Switch the override for {second} from active to inactive, or from inactive to active.",
            f"Exchange the active/inactive statuses of the overrides for {first} and {second}.",
            f"If the override for {second} is active, impose the override for {first}.",
            f"If the override for {first} is active, impose the override for {second}.",
            f"If the override for {second} is active, remove the override for {first}.",
            f"If the override for {first} is active, remove the override for {second}.",
            f"If the override for {second} is active, switch the active/inactive status of {first}'s override.",
            f"If the override for {first} is active, switch the active/inactive status of {second}'s override.",
            "Keep the current intervention state unchanged.",
        ]
        return plain[action]
    paraphrases = [
        f"Bring {first}'s suggested clamp into force.", f"Bring {second}'s suggested clamp into force.",
        f"Retire the clamp on {first}; its ordinary mechanism resumes.",
        f"Retire the clamp on {second}; its ordinary mechanism resumes.",
        "Bring both suggested clamps into force.", "Let both ordinary mechanisms resume.",
        f"Reverse whether {first} is clamped, without altering its suggested constant.",
        f"Reverse whether {second} is clamped, without altering its suggested constant.",
        f"Give {first} the clamp status previously held by {second}, and give {second} the status previously held by {first}.",
        f"Provided {second} is clamped, bring {first}'s clamp into force; otherwise leave everything as is.",
        f"Provided {first} is clamped, bring {second}'s clamp into force; otherwise leave everything as is.",
        f"Provided {second} is clamped, let {first}'s ordinary mechanism resume; otherwise leave everything as is.",
        f"Provided {first} is clamped, let {second}'s ordinary mechanism resume; otherwise leave everything as is.",
        f"Provided {second} is clamped, reverse whether {first} is clamped; otherwise leave everything as is.",
        f"Provided {first} is clamped, reverse whether {second} is clamped; otherwise leave everything as is.",
        "Carry on with exactly the same clamps as before.",
    ]
    return paraphrases[action]
