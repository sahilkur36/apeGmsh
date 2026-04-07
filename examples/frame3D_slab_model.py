# openseespy script
# Model : Frame3D_slab_ops
# Source: pyGmsh / OpenSees composite
import openseespy.opensees as ops

# ──────────────────────────────────────────────────────────────
# Model builder
ops.model('basic', '-ndm', 3, '-ndf', 6)

# ──────────────────────────────────────────────────────────────
# Nodes  (93)
ops.node(1, 9.094947018e-13, 7000, 3000)
ops.node(2, 8000, 7000, 3000)
ops.node(3, 5000, 7000, 3000)
ops.node(4, 13000, 7000, 3000)
ops.node(5, 13000, -9.094947018e-13, 3000)
ops.node(6, 9.094947018e-13, 4000, 3000)
ops.node(7, 8000, 4000, 3000)
ops.node(8, 0, 0, 3000)
ops.node(9, 13000, 4000, 3000)
ops.node(10, 8000, -9.094947018e-13, 3000)
ops.node(11, 5000, -9.094947018e-13, 3000)
ops.node(12, 5000, 4000, 3000)
ops.node(13, 1250, 4000, 3000)
ops.node(14, 2500, 4000, 3000)
ops.node(15, 3750, 4000, 3000)
ops.node(16, 5000, 5500, 3000)
ops.node(17, 3750, 7000, 3000)
ops.node(18, 2500, 7000, 3000)
ops.node(19, 1250, 7000, 3000)
ops.node(20, 9.094947018e-13, 5500, 3000)
ops.node(21, 9250, 4000, 3000)
ops.node(22, 10500, 4000, 3000)
ops.node(23, 11750, 4000, 3000)
ops.node(24, 13000, 5500, 3000)
ops.node(25, 11750, 7000, 3000)
ops.node(26, 10500, 7000, 3000)
ops.node(27, 9250, 7000, 3000)
ops.node(28, 8000, 5500, 3000)
ops.node(29, 6500, 4000, 3000)
ops.node(30, 6500, 7000, 3000)
ops.node(31, 1250, -1.530808499e-13, 3000)
ops.node(32, 2500, -3.061616998e-13, 3000)
ops.node(33, 3750, -4.592425497e-13, 3000)
ops.node(34, 5000, 1333.333333, 3000)
ops.node(35, 5000, 2666.666667, 3000)
ops.node(36, 0, 2666.666667, 3000)
ops.node(37, 0, 1333.333333, 3000)
ops.node(38, 6500, -9.184850994e-13, 3000)
ops.node(39, 8000, 1333.333333, 3000)
ops.node(40, 8000, 2666.666667, 3000)
ops.node(41, 9250, -7.654042495e-13, 3000)
ops.node(42, 10500, -9.184850994e-13, 3000)
ops.node(43, 11750, -1.071565949e-12, 3000)
ops.node(44, 13000, 1333.333333, 3000)
ops.node(45, 13000, 2666.666667, 3000)
ops.node(46, 1875, 5082.531755, 3000)
ops.node(47, 3125, 5917.468245, 3000)
ops.node(48, 1908.114711, 6191.868823, 3000)
ops.node(49, 3091.885289, 4808.131177, 3000)
ops.node(50, 860.7238392, 4946.436533, 3000)
ops.node(51, 4139.276161, 6053.563467, 3000)
ops.node(52, 4017.693575, 5046.527148, 3000)
ops.node(53, 982.306425, 5953.472852, 3000)
ops.node(54, 11125, 5082.531755, 3000)
ops.node(55, 9875, 5917.468245, 3000)
ops.node(56, 9908.114711, 4808.131177, 3000)
ops.node(57, 11091.88529, 6191.868823, 3000)
ops.node(58, 8860.723839, 6053.563467, 3000)
ops.node(59, 12139.27616, 4946.436533, 3000)
ops.node(60, 12017.69357, 5953.472852, 3000)
ops.node(61, 8982.306425, 5046.527148, 3000)
ops.node(62, 5797.831633, 4797.831633, 3000)
ops.node(63, 6125, 5875, 3000)
ops.node(64, 6929.846939, 5054.846939, 3000)
ops.node(65, 7156.25, 6156.25, 3000)
ops.node(66, 3856.781042, 2000, 3000)
ops.node(67, 1884.017314, 1036.689528, 3000)
ops.node(68, 1884.017314, 2963.310472, 3000)
ops.node(69, 3125, 1032.051579, 3000)
ops.node(70, 3125, 2967.948421, 3000)
ops.node(71, 1073.235158, 2000, 3000)
ops.node(72, 2491.341805, 2000, 3000)
ops.node(73, 899.5255399, 3075.96564, 3000)
ops.node(74, 4146.356208, 3126.923018, 3000)
ops.node(75, 4146.356208, 873.0769824, 3000)
ops.node(76, 899.5255399, 924.0343604, 3000)
ops.node(77, 6914.942527, 1970.616057, 3000)
ops.node(78, 5802.223628, 2000, 3000)
ops.node(79, 6029.122402, 2973.679045, 3000)
ops.node(80, 6077.861026, 995.1026762, 3000)
ops.node(81, 7044.891056, 3042.461043, 3000)
ops.node(82, 7098.560711, 859.8104133, 3000)
ops.node(83, 11856.78104, 2000, 3000)
ops.node(84, 9884.017314, 2963.310472, 3000)
ops.node(85, 9884.017314, 1036.689528, 3000)
ops.node(86, 11125, 1032.051579, 3000)
ops.node(87, 11125, 2967.948421, 3000)
ops.node(88, 9073.235158, 2000, 3000)
ops.node(89, 10525.90661, 2000, 3000)
ops.node(90, 8899.52554, 924.0343604, 3000)
ops.node(91, 12146.35621, 873.0769824, 3000)
ops.node(92, 12146.35621, 3126.923018, 3000)
ops.node(93, 8899.52554, 3075.96564, 3000)

# ──────────────────────────────────────────────────────────────
# nDMaterials  (0)

# ──────────────────────────────────────────────────────────────
# uniaxialMaterials  (0)

# ──────────────────────────────────────────────────────────────
# Sections  (1)
ops.section('ElasticMembranePlateSection', 1, 30000000000.0, 0.2, 200.0, 0.0)  # SlabSection

# ──────────────────────────────────────────────────────────────
# GeomTransfs  (0)

# ──────────────────────────────────────────────────────────────
# Elements  (154)
ops.element('ShellMITC3', 1, 47, 48, 46, 1)  # slab
ops.element('ShellMITC3', 2, 46, 49, 47, 1)  # slab
ops.element('ShellMITC3', 3, 14, 46, 13, 1)  # slab
ops.element('ShellMITC3', 4, 18, 47, 17, 1)  # slab
ops.element('ShellMITC3', 5, 14, 49, 46, 1)  # slab
ops.element('ShellMITC3', 6, 18, 48, 47, 1)  # slab
ops.element('ShellMITC3', 7, 3, 51, 16, 1)  # slab
ops.element('ShellMITC3', 8, 6, 50, 20, 1)  # slab
ops.element('ShellMITC3', 9, 16, 52, 12, 1)  # slab
ops.element('ShellMITC3', 10, 20, 53, 1, 1)  # slab
ops.element('ShellMITC3', 11, 49, 52, 47, 1)  # slab
ops.element('ShellMITC3', 12, 48, 53, 46, 1)  # slab
ops.element('ShellMITC3', 13, 13, 50, 6, 1)  # slab
ops.element('ShellMITC3', 14, 17, 51, 3, 1)  # slab
ops.element('ShellMITC3', 15, 47, 52, 51, 1)  # slab
ops.element('ShellMITC3', 16, 46, 53, 50, 1)  # slab
ops.element('ShellMITC3', 17, 12, 52, 15, 1)  # slab
ops.element('ShellMITC3', 18, 1, 53, 19, 1)  # slab
ops.element('ShellMITC3', 19, 15, 49, 14, 1)  # slab
ops.element('ShellMITC3', 20, 19, 48, 18, 1)  # slab
ops.element('ShellMITC3', 21, 47, 51, 17, 1)  # slab
ops.element('ShellMITC3', 22, 46, 50, 13, 1)  # slab
ops.element('ShellMITC3', 23, 15, 52, 49, 1)  # slab
ops.element('ShellMITC3', 24, 19, 53, 48, 1)  # slab
ops.element('ShellMITC3', 25, 51, 52, 16, 1)  # slab
ops.element('ShellMITC3', 26, 50, 53, 20, 1)  # slab
ops.element('ShellMITC3', 27, 55, 56, 54, 1)  # slab
ops.element('ShellMITC3', 28, 54, 57, 55, 1)  # slab
ops.element('ShellMITC3', 29, 23, 54, 22, 1)  # slab
ops.element('ShellMITC3', 30, 27, 55, 26, 1)  # slab
ops.element('ShellMITC3', 31, 54, 56, 22, 1)  # slab
ops.element('ShellMITC3', 32, 55, 57, 26, 1)  # slab
ops.element('ShellMITC3', 33, 28, 58, 2, 1)  # slab
ops.element('ShellMITC3', 34, 24, 59, 9, 1)  # slab
ops.element('ShellMITC3', 35, 7, 61, 28, 1)  # slab
ops.element('ShellMITC3', 36, 4, 60, 24, 1)  # slab
ops.element('ShellMITC3', 37, 54, 60, 57, 1)  # slab
ops.element('ShellMITC3', 38, 55, 61, 56, 1)  # slab
ops.element('ShellMITC3', 39, 9, 59, 23, 1)  # slab
ops.element('ShellMITC3', 40, 2, 58, 27, 1)  # slab
ops.element('ShellMITC3', 41, 59, 60, 54, 1)  # slab
ops.element('ShellMITC3', 42, 58, 61, 55, 1)  # slab
ops.element('ShellMITC3', 43, 25, 60, 4, 1)  # slab
ops.element('ShellMITC3', 44, 21, 61, 7, 1)  # slab
ops.element('ShellMITC3', 45, 22, 56, 21, 1)  # slab
ops.element('ShellMITC3', 46, 26, 57, 25, 1)  # slab
ops.element('ShellMITC3', 47, 23, 59, 54, 1)  # slab
ops.element('ShellMITC3', 48, 27, 58, 55, 1)  # slab
ops.element('ShellMITC3', 49, 56, 61, 21, 1)  # slab
ops.element('ShellMITC3', 50, 57, 60, 25, 1)  # slab
ops.element('ShellMITC3', 51, 24, 60, 59, 1)  # slab
ops.element('ShellMITC3', 52, 28, 61, 58, 1)  # slab
ops.element('ShellMITC3', 53, 28, 64, 7, 1)  # slab
ops.element('ShellMITC3', 54, 16, 63, 3, 1)  # slab
ops.element('ShellMITC3', 55, 3, 63, 30, 1)  # slab
ops.element('ShellMITC3', 56, 7, 64, 29, 1)  # slab
ops.element('ShellMITC3', 57, 2, 65, 28, 1)  # slab
ops.element('ShellMITC3', 58, 30, 65, 2, 1)  # slab
ops.element('ShellMITC3', 59, 12, 62, 16, 1)  # slab
ops.element('ShellMITC3', 60, 29, 62, 12, 1)  # slab
ops.element('ShellMITC3', 61, 28, 65, 64, 1)  # slab
ops.element('ShellMITC3', 62, 62, 63, 16, 1)  # slab
ops.element('ShellMITC3', 63, 64, 65, 63, 1)  # slab
ops.element('ShellMITC3', 64, 62, 64, 63, 1)  # slab
ops.element('ShellMITC3', 65, 63, 65, 30, 1)  # slab
ops.element('ShellMITC3', 66, 29, 64, 62, 1)  # slab
ops.element('ShellMITC3', 67, 71, 72, 68, 1)  # slab
ops.element('ShellMITC3', 68, 67, 72, 71, 1)  # slab
ops.element('ShellMITC3', 69, 35, 66, 34, 1)  # slab
ops.element('ShellMITC3', 70, 13, 68, 14, 1)  # slab
ops.element('ShellMITC3', 71, 32, 67, 31, 1)  # slab
ops.element('ShellMITC3', 72, 68, 70, 14, 1)  # slab
ops.element('ShellMITC3', 73, 14, 70, 15, 1)  # slab
ops.element('ShellMITC3', 74, 32, 69, 67, 1)  # slab
ops.element('ShellMITC3', 75, 33, 69, 32, 1)  # slab
ops.element('ShellMITC3', 76, 37, 71, 36, 1)  # slab
ops.element('ShellMITC3', 77, 70, 72, 66, 1)  # slab
ops.element('ShellMITC3', 78, 66, 72, 69, 1)  # slab
ops.element('ShellMITC3', 79, 36, 73, 6, 1)  # slab
ops.element('ShellMITC3', 80, 8, 76, 37, 1)  # slab
ops.element('ShellMITC3', 81, 12, 74, 35, 1)  # slab
ops.element('ShellMITC3', 82, 34, 75, 11, 1)  # slab
ops.element('ShellMITC3', 83, 6, 73, 13, 1)  # slab
ops.element('ShellMITC3', 84, 31, 76, 8, 1)  # slab
ops.element('ShellMITC3', 85, 15, 74, 12, 1)  # slab
ops.element('ShellMITC3', 86, 11, 75, 33, 1)  # slab
ops.element('ShellMITC3', 87, 68, 72, 70, 1)  # slab
ops.element('ShellMITC3', 88, 69, 72, 67, 1)  # slab
ops.element('ShellMITC3', 89, 35, 74, 66, 1)  # slab
ops.element('ShellMITC3', 90, 66, 75, 34, 1)  # slab
ops.element('ShellMITC3', 91, 13, 73, 68, 1)  # slab
ops.element('ShellMITC3', 92, 68, 73, 71, 1)  # slab
ops.element('ShellMITC3', 93, 66, 74, 70, 1)  # slab
ops.element('ShellMITC3', 94, 69, 75, 66, 1)  # slab
ops.element('ShellMITC3', 95, 67, 76, 31, 1)  # slab
ops.element('ShellMITC3', 96, 71, 76, 67, 1)  # slab
ops.element('ShellMITC3', 97, 71, 73, 36, 1)  # slab
ops.element('ShellMITC3', 98, 37, 76, 71, 1)  # slab
ops.element('ShellMITC3', 99, 70, 74, 15, 1)  # slab
ops.element('ShellMITC3', 100, 33, 75, 69, 1)  # slab
ops.element('ShellMITC3', 101, 11, 80, 34, 1)  # slab
ops.element('ShellMITC3', 102, 38, 80, 11, 1)  # slab
ops.element('ShellMITC3', 103, 40, 77, 39, 1)  # slab
ops.element('ShellMITC3', 104, 29, 81, 7, 1)  # slab
ops.element('ShellMITC3', 105, 12, 79, 29, 1)  # slab
ops.element('ShellMITC3', 106, 7, 81, 40, 1)  # slab
ops.element('ShellMITC3', 107, 35, 79, 12, 1)  # slab
ops.element('ShellMITC3', 108, 77, 82, 39, 1)  # slab
ops.element('ShellMITC3', 109, 10, 82, 38, 1)  # slab
ops.element('ShellMITC3', 110, 34, 78, 35, 1)  # slab
ops.element('ShellMITC3', 111, 34, 80, 78, 1)  # slab
ops.element('ShellMITC3', 112, 77, 79, 78, 1)  # slab
ops.element('ShellMITC3', 113, 80, 82, 77, 1)  # slab
ops.element('ShellMITC3', 114, 40, 81, 77, 1)  # slab
ops.element('ShellMITC3', 115, 77, 81, 79, 1)  # slab
ops.element('ShellMITC3', 116, 39, 82, 10, 1)  # slab
ops.element('ShellMITC3', 117, 78, 80, 77, 1)  # slab
ops.element('ShellMITC3', 118, 78, 79, 35, 1)  # slab
ops.element('ShellMITC3', 119, 79, 81, 29, 1)  # slab
ops.element('ShellMITC3', 120, 38, 82, 80, 1)  # slab
ops.element('ShellMITC3', 121, 85, 89, 88, 1)  # slab
ops.element('ShellMITC3', 122, 88, 89, 84, 1)  # slab
ops.element('ShellMITC3', 123, 45, 83, 44, 1)  # slab
ops.element('ShellMITC3', 124, 42, 85, 41, 1)  # slab
ops.element('ShellMITC3', 125, 21, 84, 22, 1)  # slab
ops.element('ShellMITC3', 126, 42, 86, 85, 1)  # slab
ops.element('ShellMITC3', 127, 43, 86, 42, 1)  # slab
ops.element('ShellMITC3', 128, 84, 87, 22, 1)  # slab
ops.element('ShellMITC3', 129, 22, 87, 23, 1)  # slab
ops.element('ShellMITC3', 130, 39, 88, 40, 1)  # slab
ops.element('ShellMITC3', 131, 83, 89, 86, 1)  # slab
ops.element('ShellMITC3', 132, 87, 89, 83, 1)  # slab
ops.element('ShellMITC3', 133, 10, 90, 39, 1)  # slab
ops.element('ShellMITC3', 134, 40, 93, 7, 1)  # slab
ops.element('ShellMITC3', 135, 44, 91, 5, 1)  # slab
ops.element('ShellMITC3', 136, 9, 92, 45, 1)  # slab
ops.element('ShellMITC3', 137, 41, 90, 10, 1)  # slab
ops.element('ShellMITC3', 138, 7, 93, 21, 1)  # slab
ops.element('ShellMITC3', 139, 5, 91, 43, 1)  # slab
ops.element('ShellMITC3', 140, 23, 92, 9, 1)  # slab
ops.element('ShellMITC3', 141, 86, 89, 85, 1)  # slab
ops.element('ShellMITC3', 142, 84, 89, 87, 1)  # slab
ops.element('ShellMITC3', 143, 83, 91, 44, 1)  # slab
ops.element('ShellMITC3', 144, 45, 92, 83, 1)  # slab
ops.element('ShellMITC3', 145, 85, 90, 41, 1)  # slab
ops.element('ShellMITC3', 146, 88, 90, 85, 1)  # slab
ops.element('ShellMITC3', 147, 86, 91, 83, 1)  # slab
ops.element('ShellMITC3', 148, 83, 92, 87, 1)  # slab
ops.element('ShellMITC3', 149, 21, 93, 84, 1)  # slab
ops.element('ShellMITC3', 150, 84, 93, 88, 1)  # slab
ops.element('ShellMITC3', 151, 39, 90, 88, 1)  # slab
ops.element('ShellMITC3', 152, 88, 93, 40, 1)  # slab
ops.element('ShellMITC3', 153, 43, 91, 86, 1)  # slab
ops.element('ShellMITC3', 154, 87, 92, 23, 1)  # slab

# ──────────────────────────────────────────────────────────────
# Single-point constraints
# PG: 'base_supports'  —  0 nodes

# ──────────────────────────────────────────────────────────────
# Load patterns
