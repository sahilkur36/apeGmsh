# openseespy script
# Model : Frame_2D_Pushover
# Source: pyGmsh / OpenSees composite
import openseespy.opensees as ops

# ──────────────────────────────────────────────────────────────
# Model builder
ops.model('basic', '-ndm', 2, '-ndf', 3)

# ──────────────────────────────────────────────────────────────
# Nodes  (76)
ops.node(1, 0, 3000)
ops.node(2, 4000, 3000)
ops.node(3, 8000, 3000)
ops.node(4, 8000, 6000)
ops.node(5, 12000, 6000)
ops.node(6, 4000, 6000)
ops.node(7, 0, 6000)
ops.node(8, 12000, 3000)
ops.node(9, 12000, 0)
ops.node(10, 8000, 0)
ops.node(11, 0, 0)
ops.node(12, 9000, 6000)
ops.node(13, 10000, 6000)
ops.node(14, 11000, 6000)
ops.node(15, 8000, 6000)
ops.node(16, 12000, 6000)
ops.node(17, 5000, 6000)
ops.node(18, 6000, 6000)
ops.node(19, 7000, 6000)
ops.node(20, 4000, 6000)
ops.node(21, 8000, 6000)
ops.node(22, 1000, 6000)
ops.node(23, 2000, 6000)
ops.node(24, 3000, 6000)
ops.node(25, 0, 6000)
ops.node(26, 4000, 6000)
ops.node(27, 12000, 3750)
ops.node(28, 12000, 4500)
ops.node(29, 12000, 5250)
ops.node(30, 12000, 3000)
ops.node(31, 12000, 6000)
ops.node(32, 8000, 3750)
ops.node(33, 8000, 4500)
ops.node(34, 8000, 5250)
ops.node(35, 8000, 3000)
ops.node(36, 8000, 6000)
ops.node(37, 4000, 3750)
ops.node(38, 4000, 4500)
ops.node(39, 4000, 5250)
ops.node(40, 4000, 3000)
ops.node(41, 4000, 6000)
ops.node(42, 0, 3750)
ops.node(43, 0, 4500)
ops.node(44, 0, 5250)
ops.node(45, 0, 3000)
ops.node(46, 0, 6000)
ops.node(47, 9000, 3000)
ops.node(48, 10000, 3000)
ops.node(49, 11000, 3000)
ops.node(50, 8000, 3000)
ops.node(51, 12000, 3000)
ops.node(52, 12000, 750)
ops.node(53, 12000, 1500)
ops.node(54, 12000, 2250)
ops.node(55, 12000, 0)
ops.node(56, 12000, 3000)
ops.node(57, 8000, 750)
ops.node(58, 8000, 1500)
ops.node(59, 8000, 2250)
ops.node(60, 8000, 0)
ops.node(61, 8000, 3000)
ops.node(62, 0, 750)
ops.node(63, 0, 1500)
ops.node(64, 0, 2250)
ops.node(65, 0, 0)
ops.node(66, 0, 3000)
ops.node(67, 1000, 3000)
ops.node(68, 2000, 3000)
ops.node(69, 3000, 3000)
ops.node(70, 0, 3000)
ops.node(71, 4000, 3000)
ops.node(72, 5000, 3000)
ops.node(73, 6000, 3000)
ops.node(74, 7000, 3000)
ops.node(75, 4000, 3000)
ops.node(76, 8000, 3000)

# ──────────────────────────────────────────────────────────────
# nDMaterials  (0)

# ──────────────────────────────────────────────────────────────
# uniaxialMaterials  (0)

# ──────────────────────────────────────────────────────────────
# Sections  (0)

# ──────────────────────────────────────────────────────────────
# GeomTransfs  (2)
ops.geomTransf('PDelta', 1)  # ColTransf
ops.geomTransf('Linear', 2)  # BeamTransf

# ──────────────────────────────────────────────────────────────
# Elements  (52)
ops.element('elasticBeamColumn', 1, 76, 32, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 2, 32, 33, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 3, 33, 34, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 4, 34, 36, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 5, 70, 42, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 6, 42, 43, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 7, 43, 44, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 8, 44, 46, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 9, 60, 57, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 10, 57, 58, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 11, 58, 59, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 12, 59, 76, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 13, 65, 62, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 14, 62, 63, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 15, 63, 64, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 16, 64, 70, 640000, np.float64(25742.960202742808), 34133333333.333332, 1)  # C80x80
ops.element('elasticBeamColumn', 17, 56, 27, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 18, 27, 28, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 19, 28, 29, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 20, 29, 31, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 21, 75, 37, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 22, 37, 38, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 23, 38, 39, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 24, 39, 41, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 25, 55, 52, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 26, 52, 53, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 27, 53, 54, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 28, 54, 56, 160000, np.float64(25742.960202742808), 2133333333.3333333, 1)  # C40x40
ops.element('elasticBeamColumn', 29, 36, 12, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 30, 12, 13, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 31, 13, 14, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 32, 14, 31, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 33, 41, 17, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 34, 17, 18, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 35, 18, 19, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 36, 19, 36, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 37, 46, 22, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 38, 22, 23, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 39, 23, 24, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 40, 24, 41, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 41, 76, 47, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 42, 47, 48, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 43, 48, 49, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 44, 49, 56, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 45, 70, 67, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 46, 67, 68, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 47, 68, 69, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 48, 69, 75, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 49, 75, 72, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 50, 72, 73, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 51, 73, 74, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50
ops.element('elasticBeamColumn', 52, 74, 76, 150000, np.float64(25742.960202742808), 3125000000.0, 2)  # V30x50

# ──────────────────────────────────────────────────────────────
# Single-point constraints
# PG: 'supports'  —  3 nodes
ops.fix(55, 1, 1, 1)
ops.fix(60, 1, 1, 1)
ops.fix(65, 1, 1, 1)

# ──────────────────────────────────────────────────────────────
# Load patterns
ops.pattern('Plain', 1, 'Linear')  # 'Pushover'
# PG: 'load_1'  —  1 nodes
ops.load(70, 1000, 0, 0)
# PG: 'load_2'  —  1 nodes
ops.load(46, 2000, 0, 0)
