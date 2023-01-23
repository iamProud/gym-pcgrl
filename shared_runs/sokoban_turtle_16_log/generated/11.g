world 	{  }
robo (worldTranslationRotation) 	{  Q:[-1, 1, 0, -1, 0, 0, 0] }
base (robo) 	{  Q:[0, 0, 0.2, -1, 0, 0, 0], shape:ssBox, size:[0.75, 0.75, 0.1, 0.05], color:[0.5, 0.5, 0.5], contact:-10, mass:1 }
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
worldTranslationRotation (world) 	{ , joint:transXY, limits:[-10, 10, -10, 10] }
block0-0 (world) 	{  Q:[-6, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-1 (world) 	{  Q:[-6, -5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-2 (world) 	{  Q:[-6, -4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-3 (world) 	{  Q:[-6, -3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-4 (world) 	{  Q:[-6, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-5 (world) 	{  Q:[-6, -1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-6 (world) 	{  Q:[-6, 0, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-7 (world) 	{  Q:[-6, 1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-8 (world) 	{  Q:[-6, 2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-9 (world) 	{  Q:[-6, 3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-10 (world) 	{  Q:[-6, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block0-11 (world) 	{  Q:[-6, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block1-0 (world) 	{  Q:[-5, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block1-10 (world) 	{  Q:[-5, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block1-11 (world) 	{  Q:[-5, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block2-0 (world) 	{  Q:[-4, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
goal2-7 (world) 	{  Q:[-4, 1, 0.001, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.95, 0.95, 0.004, 0.002], contact:-1 }
block2-11 (world) 	{  Q:[-4, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block3-0 (world) 	{  Q:[-3, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
box3-4 (world) 	{  Q:[-3, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.5, 0.5, 1, 0.02], contact:-1 }
block3-10 (world) 	{  Q:[-3, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block3-11 (world) 	{  Q:[-3, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block4-0 (world) 	{  Q:[-2, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block4-7 (world) 	{  Q:[-2, 1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block4-8 (world) 	{  Q:[-2, 2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block4-11 (world) 	{  Q:[-2, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block5-0 (world) 	{  Q:[-1, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block5-2 (world) 	{  Q:[-1, -4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block5-3 (world) 	{  Q:[-1, -3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block5-11 (world) 	{  Q:[-1, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block6-0 (world) 	{  Q:[0, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block6-3 (world) 	{  Q:[0, -3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
box6-4 (world) 	{  Q:[0, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.5, 0.5, 1, 0.02], contact:-1 }
block6-5 (world) 	{  Q:[0, -1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block6-6 (world) 	{  Q:[0, 0, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block6-7 (world) 	{  Q:[0, 1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block6-11 (world) 	{  Q:[0, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block7-0 (world) 	{  Q:[1, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block7-5 (world) 	{  Q:[1, -1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
goal7-7 (world) 	{  Q:[1, 1, 0.001, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.95, 0.95, 0.004, 0.002], contact:-1 }
block7-11 (world) 	{  Q:[1, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block8-0 (world) 	{  Q:[2, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block8-4 (world) 	{  Q:[2, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
goal8-5 (world) 	{  Q:[2, -1, 0.001, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.95, 0.95, 0.004, 0.002], contact:-1 }
box8-9 (world) 	{  Q:[2, 3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[0.5, 0.5, 1, 0.02], contact:-1 }
block8-10 (world) 	{  Q:[2, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block8-11 (world) 	{  Q:[2, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block9-0 (world) 	{  Q:[3, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block9-2 (world) 	{  Q:[3, -4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block9-10 (world) 	{  Q:[3, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block9-11 (world) 	{  Q:[3, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block10-0 (world) 	{  Q:[4, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block10-4 (world) 	{  Q:[4, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block10-5 (world) 	{  Q:[4, -1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block10-7 (world) 	{  Q:[4, 1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block10-11 (world) 	{  Q:[4, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-0 (world) 	{  Q:[5, -6, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-1 (world) 	{  Q:[5, -5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-2 (world) 	{  Q:[5, -4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-3 (world) 	{  Q:[5, -3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-4 (world) 	{  Q:[5, -2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-5 (world) 	{  Q:[5, -1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-6 (world) 	{  Q:[5, 0, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-7 (world) 	{  Q:[5, 1, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-8 (world) 	{  Q:[5, 2, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-9 (world) 	{  Q:[5, 3, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-10 (world) 	{  Q:[5, 4, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }
block11-11 (world) 	{  Q:[5, 5, 0.5, -1, 0, 0, 0], joint:rigid, shape:ssBox, size:[1, 1, 1, 0.02], contact:1, mass:100 }

