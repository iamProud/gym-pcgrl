floor 	{  X:[-0.5, -0.5, -0.05, 1, 0, 0, 0], shape:ssBox, size:[12, 12, 0.1, 0] }
robo (worldTranslationRotation) 	{  Q:[-3.5, -4.5, 0.05, -1, 0, 0, 0] }
base (robo) 	{  Q:[0, 0, 0.25, -1, 0, 0, 0], shape:ssBox, size:[0.75, 0.75, 0.1, 0.05], color:[0.5, 0.5, 0.5], contact:-10, mass:1 }
wheel1 (base) 	{  Q:[0.2, 0.2, -0.1, 1, 0, 0, 0], shape:sphere, size:[0, 0, 0, 0.1] }
wheel2 (base) 	{  Q:[-0.2, 0.2, -0.1, 1, 0, 0, 0], shape:sphere, size:[0, 0, 0, 0.1] }
wheel3 (base) 	{  Q:[0.2, -0.2, -0.1, 1, 0, 0, 0], shape:sphere, size:[0, 0, 0, 0.1] }
wheel4 (base) 	{  Q:[-0.2, -0.2, -0.1, 1, 0, 0, 0], shape:sphere, size:[0, 0, 0, 0.1] }
body (bodyRotation) 	{  Q:[0, 0, 0.35, 1, 0, 0, 0], shape:cylinder, size:[0.7, 0.3], contact:-10 }
bodyRotation_pre (base) 	{  Q:[0, 0, 0.05, 1, 0, 0, 0] }
bodyRotation (bodyRotation_pre) 	{ , joint:hingeZ, limits:[-10, 10] }
head (body) 	{  Q:[0, 0, 0.35, 1, 0, 0, 0], shape:sphere, size:[0, 0, 0, 0.3] }
armL (body) 	{  Q:[0.225, 0.3, 0.05, 1, 0, 0, 0], shape:ssBox, size:[0.45, 0.1, 0.15, 0.03], contact:-4, mass:1 }
armR (body) 	{  Q:[0.225, -0.3, 0.05, 1, 0, 0, 0], shape:ssBox, size:[0.45, 0.1, 0.15, 0.03], contact:-4, mass:1 }
worldTranslationRotation (floor) 	{ , joint:transXY, limits:[-10, 10, -10, 10] }
block0-0 (floor) 	{  Q:[-5.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-1 (floor) 	{  Q:[-5.5, -4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block0-2 (floor) 	{  Q:[-5.5, -3.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-3 (floor) 	{  Q:[-5.5, -2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block0-4 (floor) 	{  Q:[-5.5, -1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-5 (floor) 	{  Q:[-5.5, -0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block0-6 (floor) 	{  Q:[-5.5, 0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-7 (floor) 	{  Q:[-5.5, 1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block0-8 (floor) 	{  Q:[-5.5, 2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-9 (floor) 	{  Q:[-5.5, 3.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block0-10 (floor) 	{  Q:[-5.5, 4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block0-11 (floor) 	{  Q:[-5.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block1-0 (floor) 	{  Q:[-4.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block1-11 (floor) 	{  Q:[-4.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block2-0 (floor) 	{  Q:[-3.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block2-6 (floor) 	{  Q:[-3.5, 0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block2-8 (floor) 	{  Q:[-3.5, 2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block2-11 (floor) 	{  Q:[-3.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block3-0 (floor) 	{  Q:[-2.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block3-1 (floor) 	{  Q:[-2.5, -4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
goal3-2 (floor) 	{  Q:[-2.5, -3.5, 0.051, -1, 0, 0, 0], shape:ssBox, size:[0.95, 0.95, 0.004, 0.002], color:[0, 1, 0] }
block3-4 (floor) 	{  Q:[-2.5, -1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
goal3-9 (floor) 	{  Q:[-2.5, 3.5, 0.051, -1, 0, 0, 0], shape:ssBox, size:[0.95, 0.95, 0.004, 0.002], color:[0, 1, 0] }
block3-11 (floor) 	{  Q:[-2.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block4-0 (floor) 	{  Q:[-1.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block4-4 (floor) 	{  Q:[-1.5, -1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block4-11 (floor) 	{  Q:[-1.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block5-0 (floor) 	{  Q:[-0.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block5-8 (floor) 	{  Q:[-0.5, 2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block5-10 (floor) 	{  Q:[-0.5, 4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block5-11 (floor) 	{  Q:[-0.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block6-0 (floor) 	{  Q:[0.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block6-3 (floor) 	{  Q:[0.5, -2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
box6-6 (floor) 	{  Q:[0.5, 0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[0.5, 0.5, 1, 0.02], color:[0.6, 0.3, 0] }
block6-11 (floor) 	{  Q:[0.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block7-0 (floor) 	{  Q:[1.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block7-3 (floor) 	{  Q:[1.5, -2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block7-11 (floor) 	{  Q:[1.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block8-0 (floor) 	{  Q:[2.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block8-6 (floor) 	{  Q:[2.5, 0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
box8-7 (floor) 	{  Q:[2.5, 1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[0.5, 0.5, 1, 0.02], color:[0.6, 0.3, 0] }
block8-10 (floor) 	{  Q:[2.5, 4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block8-11 (floor) 	{  Q:[2.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block9-0 (floor) 	{  Q:[3.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block9-3 (floor) 	{  Q:[3.5, -2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block9-11 (floor) 	{  Q:[3.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block10-0 (floor) 	{  Q:[4.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block10-11 (floor) 	{  Q:[4.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-0 (floor) 	{  Q:[5.5, -5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-1 (floor) 	{  Q:[5.5, -4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block11-2 (floor) 	{  Q:[5.5, -3.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-3 (floor) 	{  Q:[5.5, -2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block11-4 (floor) 	{  Q:[5.5, -1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-5 (floor) 	{  Q:[5.5, -0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block11-6 (floor) 	{  Q:[5.5, 0.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-7 (floor) 	{  Q:[5.5, 1.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block11-8 (floor) 	{  Q:[5.5, 2.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-9 (floor) 	{  Q:[5.5, 3.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }
block11-10 (floor) 	{  Q:[5.5, 4.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[1, 0.3, 0.2], contact:1, mass:100 }
block11-11 (floor) 	{  Q:[5.5, 5.5, 0.55, -1, 0, 0, 0], shape:ssBox, size:[1, 1, 1, 0.02], color:[0.8, 0.1, 0.1], contact:1, mass:100 }

