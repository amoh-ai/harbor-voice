"""EVA's own code, vendored where harbor-voice runs it instead of EVA.

``perturbation.py`` is EVA's ``AudioPerturbator``
(``src/eva/user_simulator/perturbation.py`` at ServiceNow/eva@039efa0a), MIT
License, Copyright (c) 2026 ServiceNow (see ``LICENSE`` here). It runs in ``main``,
where the call loop is, because EVA applies it to the caller's audio on its
way to the agent, and the sidecar is not on that path. It is EVA's file but
for the one line that says where the noise WAVs are; it is never linted or
reformatted, and ``tests/test_eva_perturbation.py`` checks it against the copy
in EVA's image and its output against EVA's (goldens).
"""
