# Security

Please report a vulnerability privately, through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**. Don't open a public
issue for it.

Include what is affected, how to reproduce it, and the version
(`pip show harbor-voice`). We aim to reply within a week.

Only the latest release gets fixes.

harbor-voice calls provider APIs with keys you supply through the environment.
It never writes them to trajectories, logs or task files; if you find a path
where a key leaks, that is a vulnerability worth reporting.
