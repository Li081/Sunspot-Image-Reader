Detection pipeline:

Sunspots are detected with an local-background threshold.

1. Limb trimming: thin perimeter ring is removed the solar disk. Steep brightness drop at the limb would otherwise be mistaken for spots.
2. Photosphere background:  a smoothed map of the normal photosphere brightness is estimated across the disk. Only pixels on the disk are used.
3. Background refinement: dark candidate pixels are excluded and the background is estimated again so that large spot groups don't lower their own reference level.
4. Spot classification: a pixel counts as a spot if it is darker than 85% of its local background.

Spot coverage is then calculated as total spot area ÷ solar disk area × 100. Only spots of at least 10 pixels are counted.
