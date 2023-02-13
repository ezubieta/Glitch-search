# Glitch-search
Looking for mini-glitches with timing residuals

This code access to n residuals and fits F0 and F1 if asked. Then it moves one residual and fits again.

Then, it compares F0[i] with F0[i+n], and also compares F0[i] with respect to the parfile, and plots histogram for both cases.

It considers the relative jump of the frequency to report an alert of a glitch depending on the selected thresh.

