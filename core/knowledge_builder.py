"""
知识树 V4 — 每个枝干自带语义向量，10 层深度计算机科学知识体系
每个非叶子节点都通过自身内容描述生成语义向量，渗透时逐层对比权重
"""
from core.tree import SemanticKnowledgeTree


def _add_leaf(tree, parent_id, leaf_id, title, content):
    """添加叶子节点（内置 content 作为 data_pointer 和向量源）"""
    return tree.add_leaf(
        parent_id=parent_id,
        leaf_id=leaf_id,
        title=title,
        content=content,
        data_pointer={
            "title": title,
            "uri": f"knowledge://{leaf_id}",
            "content_preview": content[:200],
        },
    )


def _add_node(tree, parent_id, node_id, name, content):
    """添加枝干节点，自带语义内容用于编码向量"""
    vector = tree.encoder.encode(content)
    return tree.add_node(
        parent_id=parent_id,
        node_id=node_id,
        name=name,
        vector=vector,
        metadata={"description": content[:200]},
    )


def build_v4(encoder=None) -> SemanticKnowledgeTree:
    """构建 v4 知识树 — 每个枝干自带语义向量，10 层深度"""
    tree = SemanticKnowledgeTree(encoder=encoder)

    # ── L1 三大主干（自带语义） ──
    # 根节点直接使用编码器向量化
    tree.root.vector = tree.encoder.encode("全球知识体系，分为自然科学、工程技术和人文社科三大领域")

    math_sci = _add_node(tree, "root", "natural_sciences", "自然科学",
        "自然科学是研究自然现象和规律的学科群，包括数学、物理、化学、生物学、地球科学等基础科学。"
        "通过观察、实验和理论推导揭示自然界的本质规律，为工程技术提供理论基础。")
    tech = _add_node(tree, "root", "engineering", "工程技术",
        "工程技术是将科学原理应用于实际生产和问题解决的学科领域。"
        "涵盖计算机科学、电子工程、机械工程、土木工程等应用学科。"
        "核心特征是实践性、创新性和系统性，推动社会技术进步和产业升级。")
    hum = _add_node(tree, "root", "humanities", "人文社科",
        "人文社科研究人类社会、文化和精神现象，包括历史、文学、哲学、社会学、心理学、政治学等。"
        "探索人类文明的演变、思想的发展和社会运行的规律，培养批判性思维和人文素养。")

    # ── 人文社科（精简） ──
    _add_leaf(tree, hum.node_id, "hum_history", "历史",
        "历史学通过史料研究人类过去。包括政治史、经济史、文化史、科技史等分支。司马迁《史记》为中国第一部纪传体通史。")
    _add_leaf(tree, hum.node_id, "hum_lit", "文学",
        "文学是以语言文字为工具反映社会生活的艺术。主要体裁：诗歌、小说、散文、戏剧。")
    _add_leaf(tree, hum.node_id, "hum_philo", "哲学",
        "哲学探讨存在、知识、价值、理性等根本问题。主要分支：形而上学、认识论、伦理学、美学。")

    # ── 自然科学（精简） ──
    math = _add_node(tree, math_sci.node_id, "math_v4", "数学",
        "数学研究数量、结构、变化和空间的学科，包括代数、几何、微积分、概率统计、离散数学等分支。"
        "数学是自然科学的语言，也是计算机科学的理论基础，从基础算术到抽象代数构成严密的逻辑体系。")
    algebra = _add_node(tree, math.node_id, "algebra_v4", "代数",
        "代数是数学中研究运算规则和结构的核心分支，包括方程求解、线性代数、抽象代数等。"
        "一元二次方程求根公式和韦达定理是代数的基本成果，广泛应用于数学建模和工程计算。")
    _add_leaf(tree, algebra.node_id, "quad_formula_v4", "一元二次方程求根公式",
        "一元二次方程 ax²+bx+c=0 (a≠0) 的求根公式为 x = [-b ± √(b²-4ac)]/(2a)。Δ=b²-4ac 为判别式。")
    _add_leaf(tree, algebra.node_id, "vieta_v4", "韦达定理",
        "韦达定理：ax²+bx+c=0 的两根 x₁,x₂ 满足 x₁+x₂=-b/a, x₁·x₂=c/a。")
    calc = _add_node(tree, math.node_id, "calc_v4", "微积分",
        "微积分研究函数的变化率和累积效应，包括极限、导数、积分和微分方程。"
        "牛顿和莱布尼茨各自独立创立微积分，是近代数学最伟大的成就之一。")
    _add_leaf(tree, calc.node_id, "derivative_v4", "导数",
        "导数 f'(x)=lim_{h→0}[f(x+h)-f(x)]/h 表示函数在某点的瞬时变化率。")
    _add_leaf(tree, calc.node_id, "integral_v4", "积分",
        "积分是微分的逆运算。定积分 ∫ₐᵇ f(x)dx 表示曲线与 x 轴间的面积。")

    # ═══════════════════════════════════════════
    # L2: 计算机科学（主领域）
    # ═══════════════════════════════════════════
    cs = _add_node(tree, tech.node_id, "cs_v4", "计算机科学",
        "计算机科学研究信息处理和计算的原理、方法和应用。"
        "涵盖计算机理论、编程语言、算法与数据结构、操作系统、计算机网络、数据库、软件工程、"
        "人工智能、机器学习、安全性等子领域。计算机科学是现代信息技术的核心驱动力。")

    # ═══════════════════════════════════════════════════════════
    # 1. 计算机理论 (L3)
    # ═══════════════════════════════════════════════════════════
    cb_theory = _add_node(tree, cs.node_id, "cb_theory_v4", "计算机理论",
        "计算机理论研究计算的本质、限制和可能性，包括信息论、计算理论和数字逻辑。"
        "它为所有计算机科学提供形式化基础，揭示可计算性的边界和信息的本质度量。")

    info_theory = _add_node(tree, cb_theory.node_id, "info_theory_v4", "信息论",
        "信息论由克劳德·香农创立，研究信息的度量、存储和通信。核心概念包括信息熵、互信息和KL散度。"
        "信息论为数据压缩、通信编码和机器学习提供了理论基础。")
    _add_leaf(tree, info_theory.node_id, "entropy_v4", "信息熵",
        "信息熵 H(X) = -∑p(x)log₂p(x) 度量随机变量的不确定性，单位为比特。熵越高，信息量越大。")
    _add_leaf(tree, info_theory.node_id, "mutual_info_v4", "互信息",
        "互信息 I(X;Y) = H(X)-H(X|Y) 度量两个随机变量共享的信息量。在特征选择中广泛应用。")
    _add_leaf(tree, info_theory.node_id, "kl_divergence_v4", "KL散度",
        "KL散度 D_KL(P||Q) = ∑P(x)log(P(x)/Q(x)) 度量两个概率分布的差异，非对称。")

    comp_theory = _add_node(tree, cb_theory.node_id, "comp_theory_v4", "计算理论",
        "计算理论研究计算的数学模型和基本限制，包括图灵机、形式语言和计算复杂性。"
        "P vs NP 问题是计算机科学最重要的未解决问题之一。")
    _add_leaf(tree, comp_theory.node_id, "turing_machine_v4", "图灵机",
        "图灵机是计算理论中的抽象模型，由一个无限长的纸带和一个读写头组成。可模拟任何计算机算法。")
    _add_leaf(tree, comp_theory.node_id, "p_vs_np_v4", "P与NP问题",
        "P类问题可在多项式时间内求解，NP类问题可在多项式时间内验证。P=NP? 是计算机科学最重要的开放问题。")
    _add_leaf(tree, comp_theory.node_id, "finite_automata_v4", "有限自动机",
        "有限自动机是只有有限个状态的抽象计算模型，有DFA和NFA两种。正则表达式等价于有限自动机。")

    digi_logic = _add_node(tree, cb_theory.node_id, "digi_logic_v4", "数字逻辑",
        "数字逻辑研究用电子电路实现布尔运算的原理，包括布尔代数、逻辑门和触发器。"
        "数字逻辑是计算机硬件的基础，从简单的逻辑门到复杂的CPU都是数字逻辑的工程实现。")
    _add_leaf(tree, digi_logic.node_id, "boolean_algebra_v4", "布尔代数",
        "布尔代数基本运算：与(AND)、或(OR)、非(NOT)。德摩根定律可将与或互转。")
    _add_leaf(tree, digi_logic.node_id, "gate_circuits_v4", "逻辑门电路",
        "基本逻辑门：与门(AND)、或门(OR)、非门(NOT)、与非门(NAND)。与非门是通用门。")
    _add_leaf(tree, digi_logic.node_id, "flip_flops_v4", "触发器和寄存器",
        "触发器是存储1位信息的基本单元。多个D触发器并联构成寄存器，是CPU寄存器的基础。")

    # ═══════════════════════════════════════════════════════════
    # 2. 编程范式 (L3)
    # ═══════════════════════════════════════════════════════════
    paradigms = _add_node(tree, cs.node_id, "paradigms_v4", "编程范式",
        "编程范式是编程的基本风格和方法论，包括面向过程、面向对象、函数式和声明式等。"
        "每种范式提供了不同的抽象方式和思维模型，理解多种范式有助于写出更好的代码。")

    procedural = _add_node(tree, paradigms.node_id, "procedural_v4", "面向过程",
        "面向过程编程以函数或过程为基本组织单位，强调顺序-选择-循环三种基本结构。"
        "C语言是面向过程编程的典型代表，适合系统级编程和性能敏感场景。")
    _add_leaf(tree, procedural.node_id, "sequence_selection_iteration_v4", "顺序-选择-循环",
        "面向过程编程的三种基本结构：顺序执行、条件选择(if-else)、循环(while/for)。")
    _add_leaf(tree, procedural.node_id, "modular_programming_v4", "模块化编程",
        "模块化编程将程序分解为独立的模块或函数，接口与实现分离，降低耦合。")
    _add_leaf(tree, procedural.node_id, "top_down_design_v4", "自顶向下设计",
        "自顶向下设计将复杂问题逐层分解为更简单的子问题，逐步细化每个模块。")

    oop = _add_node(tree, paradigms.node_id, "oop_v4", "面向对象",
        "面向对象编程以类和对象为基本组织单位，核心特性包括封装、继承和多态。"
        "Java和C++是面向对象编程的典型代表，广泛应用于大型软件系统开发。")
    _add_leaf(tree, oop.node_id, "encapsulation_v4", "封装",
        "封装将数据和操作数据的方法绑定在一起，通过访问控制(public/protected/private)实现信息隐藏。")
    _add_leaf(tree, oop.node_id, "inheritance_v4", "继承",
        "继承允许子类复用父类的属性和方法。单继承(Java) vs 多继承(C++)。")
    _add_leaf(tree, oop.node_id, "polymorphism_v4", "多态",
        "多态指同一操作在不同对象上有不同行为。编译时多态(重载)和运行时多态(重写)。")

    functional = _add_node(tree, paradigms.node_id, "functional_v4", "函数式编程",
        "函数式编程以纯函数为核心，强调不可变性和无副作用。核心概念包括纯函数、高阶函数、柯里化和Monad。"
        "Haskell、Scala、Clojure等语言支持函数式编程，现代语言(JS/Python/Kotlin)也吸收了许多函数式特性。")
    _add_leaf(tree, functional.node_id, "pure_functions_v4", "纯函数",
        "纯函数相同输入总是返回相同输出，无副作用。好处：可缓存、可测试、可并行。")
    _add_leaf(tree, functional.node_id, "immutability_v4", "不可变性",
        "数据创建后不可修改，更新操作返回新对象。避免副作用简化并发。")
    _add_leaf(tree, functional.node_id, "higher_order_v4", "高阶函数",
        "高阶函数接受函数作为参数或返回函数。map/filter/reduce是典型例子。")
    _add_leaf(tree, functional.node_id, "currying_v4", "柯里化",
        "柯里化将多参数函数转为单参数函数链：f(a,b,c) → f(a)(b)(c)。")
    _add_leaf(tree, functional.node_id, "monad_v4", "Monad",
        "Monad封装计算上下文。Maybe处理空值，Either处理错误，IO处理副作用。")

    # ═══════════════════════════════════════════════════════════
    # 3. 编程语言 — Python 深度 10 层 (L3)
    # ═══════════════════════════════════════════════════════════
    langs = _add_node(tree, cs.node_id, "languages_v4", "编程语言",
        "编程语言是人与计算机交流的形式化语言。每种语言有其设计哲学、类型系统和适用场景。"
        "主流编程语言包括Python、Java、JavaScript、C/C++、Go、Rust等。")

    py = _add_node(tree, langs.node_id, "python_v4", "Python",
        "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。"
        "由 Guido van Rossum 于 1989 年底发明，第一个公开发行版发行于 1991 年。"
        "Python 语法简洁清晰，强调代码可读性，支持多种编程范式。"
        "广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。")

    # ── Python 语言核心 (L5) ──
    py_core = _add_node(tree, py.node_id, "py_core_v4", "语言核心",
        "Python语言核心包括变量与类型系统、控制流、函数定义、字符串操作等基础语法要素。"
        "Python的设计哲学强调简洁和可读性，'Pythonic'代表符合语言习惯的高质量代码风格。")

    # 变量与类型 (L6)
    py_types = _add_node(tree, py_core.node_id, "py_types_v4", "变量与类型系统",
        "Python 是动态类型语言，变量只是对象的标签。类型系统支持基本数据类型、类型注解和鸭子类型。"
        "理解 Python 的类型系统是写出健壮代码的基础。")

    # 基本数据类型 (L7)
    py_basic_types = _add_node(tree, py_types.node_id, "py_basic_types_v4", "基本数据类型",
        "Python 基本数据类型包括整型(int)、浮点型(float)、复数(complex)、布尔型(bool)和NoneType。"
        "数值类型支持任意精度运算，是Python数据处理的基础。")
    _add_leaf(tree, py_basic_types.node_id, "py_int_v4", "整型 int",
        "Python 的 int 是任意精度整数（大整数自动扩展）。支持二进制(0b)、八进制(0o)、十六进制(0x)字面量。")
    _add_leaf(tree, py_basic_types.node_id, "py_float_v4", "浮点型 float",
        "Python 的 float 是双精度64位浮点数(IEEE 754)。浮点运算有精度误差，decimal.Decimal提供精确计算。")
    _add_leaf(tree, py_basic_types.node_id, "py_bool_none_v4", "布尔与 None",
        "bool 是 int 的子类，True/False 值为 1/0。None 表示空值。条件中 False/0/''/[]/{} 均为假。")

    # int 进阶 (L9→L10)
    py_int_sub = _add_node(tree, py_basic_types.node_id, "py_int_sub_v4", "int 进阶",
        "Python int 的进阶知识包括大整数运算、位操作和进制转换。位运算比算术运算更快，适合底层优化。")
    _add_leaf(tree, py_int_sub.node_id, "py_int_bignum_v4", "大整数运算",
        "Python int 自动处理任意大整数，可计算 2**1000000 这种天文数字。在RSA密码学中广泛应用。")
    _add_leaf(tree, py_int_sub.node_id, "py_int_bitops_v4", "位运算",
        "& 按位与、| 按位或、^ 异或、<< 左移、>> 右移。比算术运算快，用于权限标记和底层优化。")
    _add_leaf(tree, py_int_sub.node_id, "py_int_bases_v4", "进制转换",
        "bin(x)二进制、oct(x)八进制、hex(x)十六进制。int('1010',2)字符串转整数。")

    # float 进阶 (L9)
    py_float_sub = _add_node(tree, py_basic_types.node_id, "py_float_sub_v4", "float 进阶",
        "浮点数的精度问题和特殊值处理是数值计算中的关键知识点。")
    _add_leaf(tree, py_float_sub.node_id, "py_float_precision_v4", "浮点精度问题",
        "0.1+0.2≠0.3 因为二进制无法精确表示某些十进制小数。使用 decimal.Decimal 或 math.isclose。")
    _add_leaf(tree, py_float_sub.node_id, "py_float_special_v4", "特殊浮点值",
        "float('inf')无穷大、float('nan')非数字。math.isinf/isnan 判断。nan任何比较都是False。")

    # 类型系统特性 (L8)
    py_type_system = _add_node(tree, py_types.node_id, "py_type_system_v4", "类型系统特性",
        "Python 类型系统的高级特性包括鸭子类型、类型注解和变量作用域规则。"
        "mypy 等工具可在编译时检查类型错误，提升代码质量。")

    # 鸭子类型 (L9)
    py_duck_typing = _add_node(tree, py_type_system.node_id, "py_duck_typing_v4", "鸭子类型",
        "鸭子类型是Python的核心设计理念：'如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子'。"
        "Python不检查对象类型，只检查是否实现了所需方法。灵活但可能运行时出错。")
    _add_leaf(tree, py_duck_typing.node_id, "py_duck_def_v4", "鸭子类型定义",
        "如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子。实现 __iter__ 即满足 Iterable 协议。")
    _add_leaf(tree, py_duck_typing.node_id, "py_duck_proto_v4", "协议与鸭子类型",
        "collections.abc 定义了 Iterable/Sized 等抽象基类作为协议的正式定义。")

    # 类型注解 (L9)
    py_type_hints = _add_node(tree, py_type_system.node_id, "py_type_hints_v4", "类型注解",
        "类型注解为Python变量和函数参数添加类型信息，提升代码可读性和IDE支持。")
    _add_leaf(tree, py_type_hints.node_id, "py_hint_basic_v4", "基本类型注解",
        "def add(x: int, y: int) -> int: 声明类型。typing: List[int]/Dict[str,int]/Optional。")
    _add_leaf(tree, py_type_hints.node_id, "py_hint_adv_v4", "高级类型注解",
        "泛型 TypeVar、联合 Union 或 |、Callable、Literal。mypy 静态类型检查。")

    # 作用域 (L8)
    py_scopes = _add_node(tree, py_types.node_id, "py_scopes_v4", "变量作用域",
        "Python变量解析遵循LEGB规则：Local→Enclosing→Global→Built-in。")
    _add_leaf(tree, py_scopes.node_id, "py_scope_legb_v4", "LEGB 规则",
        "Python 变量解析遵循 Local → Enclosing → Global → Built-in 顺序。")

    # ── 控制流 (L6) ──
    py_flow = _add_node(tree, py_core.node_id, "py_flow_v4", "控制流",
        "Python 控制流包括条件判断(if-elif-else)和循环(for/while)结构。"
        "Python 3.10+ 还引入了 match/case 模式匹配语法。")

    # 条件判断 (L7)
    py_cond = _add_node(tree, py_flow.node_id, "py_cond_v4", "条件判断",
        "条件判断通过 if、elif、else 关键字实现分支逻辑。Python 无 switch/case。")
    _add_leaf(tree, py_cond.node_id, "py_if_statement_v4", "if 语句",
        "if condition: 判断。elif 多分支。else 兜底。三元表达式: x if cond else y。")
    _add_leaf(tree, py_cond.node_id, "py_match_v4", "模式匹配 (match)",
        "Python 3.10+ 的 match/case 支持字面值、通配符、序列解包、守卫条件等多种模式。")

    # 循环 (L7)
    py_loop = _add_node(tree, py_flow.node_id, "py_loop_v4", "循环",
        "Python 循环包括 for 循环(遍历可迭代对象)和 while 循环(条件驱动)。"
        "break/continue 控制循环流程，else 子句在循环正常结束时执行。")

    # for 循环 (L8)
    py_for = _add_node(tree, py_loop.node_id, "py_for_v4", "for 循环",
        "for 循环遍历可迭代对象。range生成等差数列，enumerate带索引遍历。"
        "for 循环的迭代原理基于迭代器协议。")
    _add_leaf(tree, py_for.node_id, "py_range_v4", "range 详解",
        "range(stop)/range(start,stop)/range(start,stop,step)。惰性求值。")
    _add_leaf(tree, py_for.node_id, "py_enumerate_v4", "enumerate 详解",
        "enumerate(iterable,start=0)返回(index,value)对，替代C风格索引循环。")
    _add_leaf(tree, py_for.node_id, "py_iter_for_v4", "for 迭代原理",
        "for x in iterable: 等价于 iter()→__next__()直到StopIteration。")

    # for 循环进阶 (L9→L10)
    py_for_detail = _add_node(tree, py_for.node_id, "py_for_detail_v4", "for 循环进阶",
        "for 循环的进阶用法包括for-else子句、zip并行迭代和reversed/sorted反向排序迭代。")
    _add_leaf(tree, py_for_detail.node_id, "py_for_else_v4", "for-else 子句",
        "for 循环的 else 子句在循环正常结束(非break)时执行。常用于搜索未找到时的兜底处理。")
    _add_leaf(tree, py_for_detail.node_id, "py_for_zip_v4", "zip 并行迭代",
        "zip(iter1, iter2) 并行迭代多个序列，取对应位置元素。zip_longest填充到最长。")
    _add_leaf(tree, py_for_detail.node_id, "py_for_reversed_v4", "reversed/sorted 迭代",
        "reversed(seq) 反向迭代。sorted(seq) 排序后迭代。reversed返回迭代器。")

    # while 循环 (L8)
    py_while = _add_node(tree, py_loop.node_id, "py_while_v4", "while 循环",
        "while 条件循环，常用于不确定次数的迭代场景。")
    _add_leaf(tree, py_while.node_id, "py_while_basic_v4", "while 基础",
        "while condition: 重复执行。break退出，continue跳过本轮。")
    _add_leaf(tree, py_while.node_id, "py_while_usage_v4", "while 典型用法",
        "while True + break 模式配合 input() 验证或队列实现生产者-消费者。")

    # ── 函数 (L6) ──
    py_func = _add_node(tree, py_core.node_id, "py_func_v4", "函数",
        "Python 函数是组织和复用代码的基本单位。支持位置参数、默认参数、可变参数(*args)和关键字参数(**kwargs)。"
        "函数可嵌套定义形成闭包，可赋值传递作为一等公民。")

    py_func_def = _add_node(tree, py_func.node_id, "py_func_def_v4", "函数定义",
        "def 语句定义函数，return 返回值。支持多种参数类型组合。")

    # 函数参数 (L9→L10)
    py_func_params = _add_node(tree, py_func_def.node_id, "py_func_params_v4", "参数传递深入",
        "Python参数传递机制和参数定义方式的深入理解是写出高质量函数的关键。")
    _add_leaf(tree, py_func_params.node_id, "py_args_kwargs_v4", "*args 和 **kwargs",
        "*args 收集多余位置参数为元组，**kwargs 收集多余关键字参数为字典。解包操作符用于函数调用。")
    _add_leaf(tree, py_func_params.node_id, "py_param_passing_v4", "参数传递机制",
        "Python 参数传递是'对象引用传递'。可变对象可被修改，不可变对象不可修改。")
    _add_leaf(tree, py_func_params.node_id, "py_keyword_only_v4", "仅关键字参数",
        "* 后的参数必须用 name=value 传递，提高代码可读性。")

    _add_leaf(tree, py_func_def.node_id, "py_def_basic_v4", "函数定义基础",
        "def 函数名(参数): 函数体。位置参数、默认参数、*args、**kwargs。")
    _add_leaf(tree, py_func_def.node_id, "py_return_v4", "返回值",
        "return 返回结果。无 return 返回 None。返回多个值本质是元组。")

    py_func_adv = _add_node(tree, py_func.node_id, "py_func_adv_v4", "函数高级特性",
        "Python的函数高级特性包括闭包、装饰器和lambda表达式。")
    _add_leaf(tree, py_func_adv.node_id, "py_lambda_v4", "lambda 表达式",
        "lambda 参数: 表达式 创建匿名函数。常用于 sorted(key=...)、map/filter。")
    _add_leaf(tree, py_func_adv.node_id, "py_closure_v4", "闭包",
        "闭包引用了外部函数变量的内部函数。即使外部函数已返回，闭包仍能访问外部变量。")
    _add_leaf(tree, py_func_adv.node_id, "py_decorator_v4", "装饰器",
        "@decorator 语法糖。装饰器是接受函数返回函数的高阶函数。functools.wraps保留元信息。")

    # ── 字符串 (L6) ──
    py_str = _add_node(tree, py_core.node_id, "py_str_v4", "字符串",
        "Python 字符串(str)是不可变的Unicode字符序列。支持索引、切片、格式化和丰富的字符串方法。")
    _add_leaf(tree, py_str.node_id, "py_str_basic_v4", "字符串基础",
        "str 不可变序列。' 和 \" 定义。f-string 是最佳格式化：f'值：{x}'。")
    _add_leaf(tree, py_str.node_id, "py_str_methods_v4", "字符串方法",
        "join/split/strip/replace/find/startswith/endswith。")
    _add_leaf(tree, py_str.node_id, "py_fstring_v4", "f-string 详解",
        "f'...{表达式}...' 嵌入变量。格式说明符：{x:.2f}保留小数，{x:>10}对齐。")

    py_str_adv = _add_node(tree, py_str.node_id, "py_str_adv_v4", "字符串进阶",
        "字符串进阶包括编码转换、高级切片和正则表达式。")
    _add_leaf(tree, py_str_adv.node_id, "py_str_bytes_encode_v4", "编码与 bytes",
        "str.encode('utf-8')→bytes，b'hello'.decode('utf-8')→str。Unicode是字符集，UTF-8是编码。")
    _add_leaf(tree, py_str_adv.node_id, "py_str_slicing_v4", "字符串切片进阶",
        "s[start:stop:step] 步长为负可反向。s[::-1] 反转字符串。")
    _add_leaf(tree, py_str_adv.node_id, "py_str_regex_v4", "字符串正则进阶",
        "re.match/re.search/re.findall/re.sub。捕获组()和非捕获组(?:)。")

    # ── 内置数据结构 (L5) ──
    py_ds = _add_node(tree, py.node_id, "py_ds_v4", "内置数据结构",
        "Python 内置了丰富的数据结构：列表(list)、字典(dict)、集合(set)和元组(tuple)。"
        "这些数据结构经过精心优化，是Python编程的基石。")

    py_list = _add_node(tree, py_ds.node_id, "py_list_v4", "列表 list",
        "列表是可变有序序列，支持索引、切片和方法操作。列表推导式是Python的标志性语法。")
    _add_leaf(tree, py_list.node_id, "py_list_basics_v4", "列表基础",
        "list 可变有序。索引从0开始，负索引从末尾。切片[start:stop:step]。")
    _add_leaf(tree, py_list.node_id, "py_list_methods_v4", "列表方法",
        "append/extend/insert/remove/pop/sort/reverse。")
    _add_leaf(tree, py_list.node_id, "py_list_comprehension_v4", "列表推导式",
        "[表达式 for 变量 in 可迭代对象 if 条件]。比 for 循环更简洁高效。")

    py_dict = _add_node(tree, py_ds.node_id, "py_dict_v4", "字典 dict",
        "字典是键值对映射结构，键必须可哈希。Python 3.7+ 保持插入顺序。")
    _add_leaf(tree, py_dict.node_id, "py_dict_basics_v4", "字典基础",
        "{} 创建。d[key] 访问，d.get(key,default) 安全访问。3.7+ 保持插入顺序。")
    _add_leaf(tree, py_dict.node_id, "py_dict_methods_v4", "字典方法",
        "keys/values/items 视图。setdefault/pop/update。| 操作符合并(3.9+)。")

    py_set = _add_node(tree, py_ds.node_id, "py_set_v4", "集合与元组",
        "集合无序不重复，支持集合运算。元组不可变有序，可用作字典键。")
    _add_leaf(tree, py_set.node_id, "py_set_basics_v4", "集合 set",
        "set 无序不重复。&交集 |并集 -差集 ^对称差集。")
    _add_leaf(tree, py_set.node_id, "py_tuple_v4", "元组 tuple",
        "tuple 不可变有序。(1,2,3)创建。可用作字典键。namedtuple具名元组。")

    # ── 面向对象 (L5) ──
    py_oop = _add_node(tree, py.node_id, "py_oop_v4", "面向对象",
        "Python 支持面向对象编程，包括类定义、继承、多态和魔术方法。"
        "Python的OOP灵活而强大，@property/@classmethod/@staticmethod等装饰器丰富类功能。")

    py_class = _add_node(tree, py_oop.node_id, "py_class_v4", "类与实例",
        "class 定义类，__init__ 构造方法。self 是实例引用。classmethod/staticmethod/property。")
    _add_leaf(tree, py_class.node_id, "py_class_def_v4", "class 定义",
        "class ClassName: 定义类。__init__构造。self实例引用。dataclass自动生成基础方法。")
    _add_leaf(tree, py_class.node_id, "py_class_advanced_v4", "类高级功能",
        "classmethod/staticmethod/property/ABC/abstractmethod/dataclass。")

    py_inherit = _add_node(tree, py_class.node_id, "py_inherit_v4", "继承与多态",
        "Python 支持单继承和多继承(Mixin)。super()调用父类。MRO决定方法搜索顺序。")
    _add_leaf(tree, py_inherit.node_id, "py_single_inherit_v4", "单继承",
        "class B(A): B继承A。super().__init__()调用父类。isinstance类型检查。")
    _add_leaf(tree, py_inherit.node_id, "py_mixin_v4", "Mixin 混入",
        "Mixin 提供特定功能的小型基类，通过多继承组合。命名约定以 Mixin 结尾。")

    py_magic = _add_node(tree, py_oop.node_id, "py_magic_v4", "魔术方法",
        "魔术方法是Python对象模型的核心，以双下划线命名(__xxx__)，实现运算符重载和协议。")
    _add_leaf(tree, py_magic.node_id, "py_magic_str_v4", "字符串表示",
        "__str__用户字符串、__repr__开发字符串、__format__格式化。")
    _add_leaf(tree, py_magic.node_id, "py_magic_container_v4", "容器方法",
        "__len__、__getitem__、__setitem__、__iter__、__contains__。")
    _add_leaf(tree, py_magic.node_id, "py_magic_ops_v4", "运算符重载",
        "__add__/__eq__/__lt__/__hash__/__call__。")

    py_except = _add_node(tree, py_oop.node_id, "py_except_v4", "异常处理",
        "Python异常处理通过try-except-else-finally结构管理运行时错误。")
    _add_leaf(tree, py_except.node_id, "py_try_except_v4", "try-except",
        "try-except-else-finally。except ValueError as e: 捕获特定异常。")
    _add_leaf(tree, py_except.node_id, "py_common_except_v4", "常见异常类型",
        "ValueError/TypeError/KeyError/IndexError/FileNotFoundError/AttributeError。")
    _add_leaf(tree, py_except.node_id, "py_custom_except_v4", "自定义异常",
        "class MyError(Exception): pass。自定义异常。")

    # ── 高级特性 (L5) ──
    py_adv = _add_node(tree, py.node_id, "py_adv_v4", "高级特性",
        "Python 高级特性包括装饰器进阶、迭代器生成器、上下文管理器和函数式工具。"
        "这些特性让Python代码更优雅、更高效。")

    py_adv_deco = _add_node(tree, py_adv.node_id, "py_adv_deco_v4", "装饰器进阶",
        "装饰器的本质是闭包，掌握其原理可写出灵活的装饰器。")
    _add_leaf(tree, py_adv_deco.node_id, "py_deco_closure_v4", "装饰器原理",
        "func = decorator(func)。闭包捕获原函数。wraps复制元信息。")
    _add_leaf(tree, py_adv_deco.node_id, "py_deco_params_v4", "带参数装饰器",
        "def decorator(args): def inner(func): ... 三层嵌套。")
    _add_leaf(tree, py_adv_deco.node_id, "py_deco_class_v4", "类装饰器",
        "类实现 __call__ 作为装饰器。适合需要持久状态的场景。")

    py_iter_gen = _add_node(tree, py_adv.node_id, "py_iter_gen_v4", "迭代器与生成器",
        "迭代器协议和生成器是Python惰性求值的核心机制，节省内存。")
    _add_leaf(tree, py_iter_gen.node_id, "py_iterator_proto_v4", "迭代器协议",
        "__iter__返回self，__next__返回元素或raise StopIteration。")
    _add_leaf(tree, py_iter_gen.node_id, "py_generator_v4", "生成器 yield",
        "yield 定义生成器函数。惰性序列，节省内存。")
    _add_leaf(tree, py_iter_gen.node_id, "py_yield_from_v4", "yield from",
        "yield from sub_gen 委托子生成器。简化生成器嵌套。")
    _add_leaf(tree, py_iter_gen.node_id, "py_gen_expression_v4", "生成器表达式",
        "(x**2 for x in range(10)) 惰性求值，比列表推导式省内存。")

    py_context = _add_node(tree, py_adv.node_id, "py_context_v4", "上下文管理器",
        "with 语句管理资源(文件/锁/连接)，通过 __enter__/__exit__ 实现。")
    _add_leaf(tree, py_context.node_id, "py_with_class_v4", "类实现上下文",
        "__enter__进入/__exit__退出(处理异常)。")
    _add_leaf(tree, py_context.node_id, "py_contextlib_v4", "contextmanager",
        "@contextmanager 装饰器函数，yield前为进入后为退出。")

    py_fp = _add_node(tree, py_adv.node_id, "py_fp_v4", "函数式工具",
        "Python 提供了map/filter/reduce、偏函数和itertools等函数式工具。")
    _add_leaf(tree, py_fp.node_id, "py_map_filter_reduce_v4", "map/filter/reduce",
        "map(func,iter)映射、filter(func,iter)过滤、reduce(func,iter)归约。")
    _add_leaf(tree, py_fp.node_id, "py_partial_v4", "偏函数 partial",
        "functools.partial固定部分参数，返回新可调用对象。")
    _add_leaf(tree, py_fp.node_id, "py_itertools_v4", "itertools 模块",
        "chain/cycle/count/product/permutations/combinations/groupby。")

    # ── 标准库 (L5) ──
    py_stdlib = _add_node(tree, py.node_id, "py_stdlib_v4", "标准库",
        "Python 标准库提供了丰富的内置模块，覆盖操作系统接口、数据处理、网络通信等。")
    _add_leaf(tree, py_stdlib.node_id, "py_json_v4", "JSON 处理",
        "json.dumps(obj,ensure_ascii=False)序列化。json.loads(str)反序列化。")
    _add_leaf(tree, py_stdlib.node_id, "py_regex_v4", "正则表达式",
        "re.search/match/findall/sub。预编译re.compile。")
    _add_leaf(tree, py_stdlib.node_id, "py_datetime_v4", "时间与日期",
        "datetime.now()/timedelta/strftime/time.sleep/timezone。")

    # ── 并发编程 (L5) ──
    py_conc = _add_node(tree, py.node_id, "py_conc_v4", "并发编程",
        "Python 支持多线程、多进程和异步编程三种并发模型，各有适用场景。")
    py_thread = _add_node(tree, py_conc.node_id, "py_thread_v4", "多线程",
        "threading模块创建线程。GIL限制CPU并行，适合I/O密集型。")
    _add_leaf(tree, py_thread.node_id, "py_threading_basic_v4", "threading 基础",
        "Thread创建/start/join。daemon守护线程。GIL限制CPU并行。")
    _add_leaf(tree, py_thread.node_id, "py_thread_sync_v4", "线程同步",
        "Lock/RLock/Semaphore/Event/Condition。")
    py_process = _add_node(tree, py_conc.node_id, "py_process_v4", "多进程",
        "multiprocessing跨过GIL，利用多核CPU。")
    _add_leaf(tree, py_process.node_id, "py_multiprocessing_v4", "multiprocessing",
        "Process/Pool.map/Queue/Pipe。不受GIL限制，适合CPU密集型。")
    py_async = _add_node(tree, py_conc.node_id, "py_async_v4", "异步编程",
        "async/await协程是单线程并发模型，适合高I/O并发。")
    _add_leaf(tree, py_async.node_id, "py_async_await_v4", "async/await",
        "async def/await/asyncio.run/asyncio.gather。")

    # ── 网络编程 (L5) ──
    py_net = _add_node(tree, py.node_id, "py_net_v4", "网络编程",
        "Python网络编程涵盖urllib内置库和requests/aiohttp等第三方库。")
    _add_leaf(tree, py_net.node_id, "py_requests_v4", "requests 库",
        "requests.get/post/put/delete。Session复用连接。最流行的HTTP库。")
    _add_leaf(tree, py_net.node_id, "py_socket_v4", "socket 底层 API",
        "socket.socket/bind/listen/accept/connect/recv/send。")

    # ── 框架生态 (L5) ──
    py_fw = _add_node(tree, py.node_id, "py_fw_v4", "框架生态",
        "Python拥有丰富的框架生态，涵盖Web开发、数据科学、机器学习等。")
    _add_leaf(tree, py_fw.node_id, "py_django_v4", "Django",
        "全栈Web框架：ORM/Admin/模板。MTV架构。DRF构建API。")
    _add_leaf(tree, py_fw.node_id, "py_fastapi_v4", "FastAPI",
        "现代异步Web框架。自动OpenAPI文档。Pydantic校验。类型注解驱动。")
    _add_leaf(tree, py_fw.node_id, "py_flask_v4", "Flask",
        "微框架：轻量灵活。Jinja2模板/Werkzeug WSGI。蓝图模块化。")
    _add_leaf(tree, py_fw.node_id, "py_numpy_v4", "NumPy",
        "科学计算基础库。ndarray高效数组。向量化运算比Python快10-100x。")
    _add_leaf(tree, py_fw.node_id, "py_pandas_v4", "Pandas",
        "DataFrame数据结构。read_csv/groupby/merge/pivot。数据科学与分析核心。")

    # ═══════════════════════════════════════════════════════════
    # 其他编程语言 (L4 Java/JS/C++)
    # ═══════════════════════════════════════════════════════════
    java = _add_node(tree, langs.node_id, "java_v4", "Java",
        "Java是静态类型、面向对象的编译型语言，JVM提供跨平台能力。"
        "Java拥有丰富的企业级生态(Spring/Hibernate)，是大型系统的常用语言。")
    java_core = _add_node(tree, java.node_id, "java_core_v4", "Java 核心",
        "Java核心包括JVM机制、OOP、泛型、集合框架和IO体系。")
    _add_leaf(tree, java_core.node_id, "java_vm_basics_v4", "JVM 基础",
        "JVM执行字节码。类加载机制。内存区域：堆/栈/方法区。GC自动回收。")
    _add_leaf(tree, java_core.node_id, "java_oop_v4", "Java OOP",
        "extends单继承，implements多接口。abstract/final。")
    _add_leaf(tree, java_core.node_id, "java_generics_v4", "泛型",
        "List<T>类型参数。通配符<? extends E>/<? super E>。类型擦除。")
    java_col = _add_node(tree, java.node_id, "java_col_v4", "集合框架",
        "Java集合框架包括List/Map/Set三大接口及其实现。")
    _add_leaf(tree, java_col.node_id, "java_list_v4", "List 系列",
        "ArrayList(数组) vs LinkedList(链表)。Vector/Stack(线程安全)。")
    _add_leaf(tree, java_col.node_id, "java_map_v4", "Map 系列",
        "HashMap(哈希)/TreeMap(红黑树)/LinkedHashMap(有序)/ConcurrentHashMap(并发)。")
    _add_leaf(tree, java_col.node_id, "java_set_v4", "Set 系列",
        "HashSet/TreeSet/LinkedHashSet。")

    js = _add_node(tree, langs.node_id, "js_v4", "JavaScript",
        "JavaScript是Web的核心编程语言，单线程非阻塞，事件驱动。ES6+带来class/箭头函数/Promise。")
    js_core = _add_node(tree, js.node_id, "js_core_v4", "JavaScript 核心",
        "JS核心包括执行上下文、作用域链、闭包、原型链和异步编程。")
    _add_leaf(tree, js_core.node_id, "js_exec_ctx_v4", "执行上下文",
        "执行上下文包含变量环境、词法环境、this绑定。执行栈管理。")
    _add_leaf(tree, js_core.node_id, "js_scope_chain_v4", "作用域链",
        "变量查找沿作用域链向上。let/const块级作用域。闭包保持外部引用。")
    _add_leaf(tree, js_core.node_id, "js_closure_v4", "闭包",
        "内部函数访问外部函数变量的能力。用于模块模式/防抖节流。")
    _add_leaf(tree, js_core.node_id, "js_prototype_v4", "原型链",
        "JS继承机制。__proto__指向prototype。属性查找沿原型链向上。")
    js_async = _add_node(tree, js_core.node_id, "js_async_v4", "异步 JS",
        "JS异步模型从回调到Promise到async/await演进。")
    _add_leaf(tree, js_async.node_id, "js_promise_v4", "Promise",
        "pending/fulfilled/rejected。then/catch链式。Promise.all/race。")
    _add_leaf(tree, js_async.node_id, "js_async_await_v4", "async/await",
        "async function返回Promise。await暂停直到Promise解决。")
    _add_leaf(tree, js_async.node_id, "js_event_loop_v4", "事件循环",
        "JS单线程。宏任务(setTimeout)和微任务(Promise)。执行顺序控制。")

    cpp = _add_node(tree, langs.node_id, "cpp_v4", "C/C++",
        "C语言是系统编程的基础语言，提供对硬件的直接控制。C++在C基础上增加OOP和模板。"
        "两者都是编译型语言，追求极致的性能和灵活的内存管理。")
    c_core = _add_node(tree, cpp.node_id, "c_core_v4", "C 核心",
        "C核心包括指针、内存管理、结构体和预处理。")
    _add_leaf(tree, c_core.node_id, "c_pointers_v4", "指针",
        "指针存储地址。*解引用/&取地址。函数指针。指针运算是C的核心能力。")
    _add_leaf(tree, c_core.node_id, "c_memory_v4", "内存管理",
        "malloc/calloc/realloc/free。栈自动回收。Valgrind检测内存错误。")
    cpp_core = _add_node(tree, cpp.node_id, "cpp_core_v4", "C++ 核心",
        "C++核心包括OOP、模板和STL。智能指针RAII管理资源。")
    _add_leaf(tree, cpp_core.node_id, "cpp_oop_v4", "C++ OOP",
        "class/public/protected/private。virtual虚函数(vtable)。纯虚函数抽象类。")
    _add_leaf(tree, cpp_core.node_id, "cpp_templates_v4", "模板",
        "template<typename T>泛型。模板特化。模板元编程编译期计算。")
    _add_leaf(tree, cpp_core.node_id, "cpp_stl_v4", "STL 标准库",
        "vector/list/map。sort/find。迭代器。函数对象与lambda。")
    _add_leaf(tree, cpp_core.node_id, "cpp_smart_ptrs_v4", "智能指针",
        "unique_ptr独占/shared_ptr共享/weak_ptr弱引用。自动释放防泄漏。")

    go = _add_node(tree, langs.node_id, "go_v4", "Go",
        "Go是Google开发的编译型语言，简洁高效，原生支持并发。goroutine和channel是Go的标志性特性。")
    _add_leaf(tree, go.node_id, "go_goroutine_v4", "Goroutine",
        "go func()启动轻量级协程。百万并发不费力。栈初始小可动态扩展。")
    _add_leaf(tree, go.node_id, "go_channel_v4", "Channel",
        "ch := make(chan int)。ch<-val发送/<-ch接收。select多路复用。")

    rust = _add_node(tree, langs.node_id, "rust_v4", "Rust",
        "Rust是Mozilla开发的系统级语言，以零成本抽象和内存安全著称。"
        "所有权系统和借用检查器在编译时确保内存安全，无需GC。")
    _add_leaf(tree, rust.node_id, "rust_ownership_v4", "所有权",
        "每个值有唯一所有者。所有权转移(move)vs借用(&/&mut)。编译期内存安全。")
    _add_leaf(tree, rust.node_id, "rust_borrow_checker_v4", "借用检查器",
        "编译时分析引用：一个可变引用或多个不可变引用，不可同时存在。")
    _add_leaf(tree, rust.node_id, "rust_trait_v4", "Trait 系统",
        "trait定义共享行为。泛型约束fn foo<T: Display>(t: T)。trait对象。")

    # ═══════════════════════════════════════════════════════════
    # 4. 算法与数据结构 (L3)
    # ═══════════════════════════════════════════════════════════
    algo = _add_node(tree, cs.node_id, "algo_v4", "算法与数据结构",
        "算法是解决问题的步骤和方法，数据结构是组织和存储数据的方式。"
        "两者是计算机科学的核心基础，直接影响程序性能和质量。")

    ds_basics = _add_node(tree, algo.node_id, "ds_basics_v4", "基础数据结构",
        "基础数据结构包括数组、链表、栈和队列，是更复杂结构的基础。")
    _add_leaf(tree, ds_basics.node_id, "array_v4", "数组",
        "连续内存线性结构。随机访问O(1)，插入删除O(n)。缓存友好。")
    _add_leaf(tree, ds_basics.node_id, "linked_list_v4", "链表",
        "节点含数据和指针。单链表/双链表/循环链表。插入删除O(1)。")
    _add_leaf(tree, ds_basics.node_id, "stack_v4", "栈",
        "后进先出LIFO。push/pop/peek O(1)。应用：函数调用/括号匹配/后退导航。")
    _add_leaf(tree, ds_basics.node_id, "queue_v4", "队列",
        "先进先出FIFO。enqueue/dequeue O(1)。应用：BFS/任务调度/消息队列。")

    sorting = _add_node(tree, algo.node_id, "sorting_v4", "排序算法",
        "排序是计算机科学中最基础和重要的算法之一。分为比较排序和非比较排序两类。")
    sort_compare = _add_node(tree, sorting.node_id, "sort_compare_v4", "比较排序",
        "比较排序基于元素间的比较操作，通用性强。")
    _add_leaf(tree, sort_compare.node_id, "quick_sort_v4", "快速排序",
        "分治：选基准(pivot)分区。平均O(n log n)，最坏O(n²)。原地不稳定。")
    _add_leaf(tree, sort_compare.node_id, "merge_sort_v4", "归并排序",
        "分治：对半分割→递归→合并。稳定O(n log n)。需O(n)空间。外部排序标准。")
    _add_leaf(tree, sort_compare.node_id, "heap_sort_v4", "堆排序",
        "建堆→交换堆顶→调整。O(n log n)原地。不稳定。无最坏退化。")
    _add_leaf(tree, sort_compare.node_id, "insertion_sort_v4", "插入排序",
        "O(n²)但小规模基本有序时O(n)。Timsort的组成部分。")
    sort_linear = _add_node(tree, sorting.node_id, "sort_linear_v4", "线性排序",
        "线性排序不通过比较，利用数据特性达到O(n)时间复杂度。")
    _add_leaf(tree, sort_linear.node_id, "counting_sort_v4", "计数排序",
        "统计出现次数→累加→输出。O(n+k)，限整数且范围不大。")
    _add_leaf(tree, sort_linear.node_id, "radix_sort_v4", "基数排序",
        "按位从低到高排序(每位稳定排序)。O(d·n)。适用整数/定长字符串。")

    search = _add_node(tree, algo.node_id, "search_v4", "搜索算法",
        "搜索算法在数据中查找目标元素，包括线性搜索、二分搜索和图搜索。")
    _add_leaf(tree, search.node_id, "linear_search_v4", "线性搜索",
        "从头到尾遍历。O(n)。最简单但低效。适用无序数据。")
    _add_leaf(tree, search.node_id, "binary_search_v4", "二分搜索",
        "每次中间比较缩小一半。O(log n)。要求已排序。Python bisect模块。")
    _add_leaf(tree, search.node_id, "bfs_v4", "广度优先搜索 BFS",
        "用队列逐层遍历。找最短路径。O(V+E)。应用：迷宫/社交网络。")
    _add_leaf(tree, search.node_id, "dfs_v4", "深度优先搜索 DFS",
        "递归或栈深入到底再回溯。O(V+E)。应用：连通分量/拓扑排序。")

    tree_algo = _add_node(tree, algo.node_id, "tree_algo_v4", "树与图算法",
        "树和图是最重要的非线性数据结构，它们的算法(遍历/最短路径/生成树)是面试和竞赛的核心。")
    binary_tree = _add_node(tree, tree_algo.node_id, "binary_tree_v4", "二叉树",
        "二叉树每个节点最多两个子节点。二叉搜索树(BST)左小右大，中序遍历得有序序列。")
    bst_node = _add_node(tree, binary_tree.node_id, "bst_v4", "二叉搜索树 BST",
        "BST是二叉树中最常用的变体。左子树<根<右子树，中序遍历有序。")
    _add_leaf(tree, bst_node.node_id, "bst_search_v4", "BST 搜索",
        "从根开始，比根小走左比根大走右。平均O(log n)，最坏O(n)。")
    _add_leaf(tree, bst_node.node_id, "bst_insert_v4", "BST 插入",
        "先搜索找到位置并插入。O(h)。不允许重复键。中序遍历得到有序。")
    bst_trav_detail = _add_node(tree, bst_node.node_id, "bst_trav_detail_v4", "遍历实现",
        "二叉树的三种深度优先遍历方式各有特点和用途。")
    _add_leaf(tree, bst_trav_detail.node_id, "bst_preorder_v4", "前序遍历实现",
        "根→左→右。递归和迭代(栈)实现。用于复制树结构。")
    _add_leaf(tree, bst_trav_detail.node_id, "bst_inorder_v4", "中序遍历实现",
        "左→根→右。BST中序得有序序列。栈模拟实现。")
    _add_leaf(tree, bst_trav_detail.node_id, "bst_postorder_v4", "后序遍历实现",
        "左→右→根。先删子节点再删父节点(释放树)。双栈实现。")

    balanced_tree = _add_node(tree, binary_tree.node_id, "balanced_tree_v4", "平衡树",
        "平衡树通过旋转或变色操作保持树的高度平衡，保证O(log n)操作时间。")
    _add_leaf(tree, balanced_tree.node_id, "avl_tree_v4", "AVL 树",
        "严格平衡：左右子树高度差≤1。四种旋转(LL/RR/LR/RL)保持平衡。")
    _add_leaf(tree, balanced_tree.node_id, "red_black_tree_v4", "红黑树",
        "近似平衡。5条性质。旋转+变色维护。Java TreeMap/nginx使用。")
    _add_leaf(tree, balanced_tree.node_id, "b_tree_v4", "B 树",
        "多路平衡树。扇出高大，降低I/O。数据库B+树索引的基础。")

    graph = _add_node(tree, tree_algo.node_id, "graph_algo_v4", "图算法",
        "图算法解决最短路径、最小生成树和拓扑排序等典型问题。")
    _add_leaf(tree, graph.node_id, "dijkstra_v4", "Dijkstra 最短路径",
        "单源最短路径(非负权)。贪心+优先队列。O(E log V)。导航核心。")
    _add_leaf(tree, graph.node_id, "floyd_v4", "Floyd-Warshall",
        "所有节点对最短路径。动态规划。O(V³)。代码极简洁。")
    _add_leaf(tree, graph.node_id, "kruskal_v4", "Kruskal 最小生成树",
        "按边权排序，不形成环的边加入。并查集检测环。O(E log E)。")

    dp = _add_node(tree, algo.node_id, "dp_v4", "动态规划",
        "动态规划将问题分解为重叠子问题。核心：状态定义→转移方程→边界条件→遍历顺序。")
    _add_leaf(tree, dp.node_id, "dp_basics_v4", "DP 基础",
        "最优子结构+重叠子问题。自顶向下(记忆化) vs 自底向上(表格)。")
    _add_leaf(tree, dp.node_id, "knapsack_v4", "背包问题",
        "0-1背包。完全背包(内循环正序)。多重背包(二进制优化)。")
    _add_leaf(tree, dp.node_id, "lcs_v4", "最长公共子序列 LCS",
        "O(mn)。DNA序列比对/文本diff。")
    _add_leaf(tree, dp.node_id, "lis_v4", "最长递增子序列 LIS",
        "O(n²)DP。二分优化O(n log n)。合唱队形/股票分析。")

    adv_ds = _add_node(tree, algo.node_id, "adv_ds_v4", "高级数据结构",
        "高级数据结构解决特定场景问题，包括并查集、线段树、字典树和布隆过滤器。")
    _add_leaf(tree, adv_ds.node_id, "union_find_v4", "并查集",
        "集合合并和查找。路径压缩+按秩合并≈O(α(n))。连通分量/Kruskal。")
    _add_leaf(tree, adv_ds.node_id, "segment_tree_v4", "线段树",
        "区间查询和单点更新O(log n)。懒标记支持区间更新。")
    _add_leaf(tree, adv_ds.node_id, "trie_v4", "字典树 Trie",
        "字符串前缀匹配。查找O(L)。自动补全/拼写检查/IP路由。")
    _add_leaf(tree, adv_ds.node_id, "bloom_filter_v4", "布隆过滤器",
        "位数组+多哈希。判断'一定不在'或'可能在'。防缓存穿透/去重。")

    # ═══════════════════════════════════════════════════════════
    # 5. 操作系统 (L3)
    # ═══════════════════════════════════════════════════════════
    os_node = _add_node(tree, cs.node_id, "os_v4", "操作系统",
        "操作系统是管理计算机硬件和软件资源的系统软件。核心功能包括进程管理、内存管理、文件系统和I/O管理。"
        "操作系统是用户和硬件之间的桥梁，常见的有Linux、Windows、macOS。")

    proc = _add_node(tree, os_node.node_id, "proc_v4", "进程与线程管理",
        "进程是资源分配的基本单位，线程是CPU调度的基本单位。操作系统通过调度算法决定哪个进程获得CPU。")
    proc_sched = _add_node(tree, proc.node_id, "proc_sched_v4", "进程调度",
        "进程调度决定就绪队列中哪个进程获得CPU，常用算法包括FCFS、SJF、RR和MLFQ。")
    sched_algo = _add_node(tree, proc_sched.node_id, "sched_algo_v4", "调度算法",
        "调度算法影响系统的吞吐量、响应时间和公平性。")
    _add_leaf(tree, sched_algo.node_id, "fcfs_v4", "先来先服务 FCFS",
        "按到达顺序。非抢占。平均等待时间长(护航效应)。批处理系统。")
    _add_leaf(tree, sched_algo.node_id, "sjf_v4", "短作业优先 SJF",
        "选最短CPU突发时间的进程。平均等待最小。无法预知。")
    _add_leaf(tree, sched_algo.node_id, "rr_v4", "时间片轮转 RR",
        "固定时间片循环执行。响应快。时间片选择关键。")

    proc_comm = _add_node(tree, proc.node_id, "proc_comm_v4", "进程间通信 IPC",
        "进程间通信方式包括管道、共享内存、信号量、消息队列和套接字。")
    _add_leaf(tree, proc_comm.node_id, "pipe_v4", "管道",
        "单向数据流。无名管道仅父子进程。命名管道mkfifo。Shell | 操作符。")
    _add_leaf(tree, proc_comm.node_id, "shm_v4", "共享内存",
        "最快IPC。多个进程映射同一物理内存。需信号量同步。")

    sync = _add_node(tree, proc.node_id, "sync_v4", "同步与互斥",
        "同步机制确保并发执行的正确性和数据一致性。")
    _add_leaf(tree, sync.node_id, "mutex_lock_v4", "互斥锁 Mutex",
        "保证同一时间只有一个线程访问临界区。P/V操作。")
    _add_leaf(tree, sync.node_id, "semaphore_v4", "信号量",
        "P(申请)/V(释放)。控制多个资源。经典问题：生产者-消费者。")
    _add_leaf(tree, sync.node_id, "deadlock_v4", "死锁",
        "四个必要条件。预防/避免/检测/恢复。银行家算法。")

    mem = _add_node(tree, os_node.node_id, "mem_v4", "内存管理",
        "内存管理负责分配和回收进程内存空间，通过虚拟内存、分页和分段机制实现。")
    mem_virt = _add_node(tree, mem.node_id, "mem_virt_v4", "虚拟内存",
        "虚拟内存将进程逻辑地址映射到物理内存，提供隔离和扩展。")
    _add_leaf(tree, mem_virt.node_id, "paging_v4", "分页机制",
        "逻辑地址→页号+偏移→页表→物理页框。TLB快表加速。4KB标准页。")
    _add_leaf(tree, mem_virt.node_id, "page_replacement_v4", "页面置换算法",
        "FIFO/LRU/Clock。缺页率影响性能。")

    fs = _add_node(tree, os_node.node_id, "fs_v4", "文件系统",
        "文件系统管理持久数据的组织和存储。VFS抽象层屏蔽底层差异。")
    vfs = _add_node(tree, fs.node_id, "vfs_v4", "虚拟文件系统 VFS",
        "VFS提供统一文件操作接口，支持ext4/XFS/Btrfs等具体文件系统。")
    _add_leaf(tree, vfs.node_id, "ext4_v4", "ext4 文件系统",
        "Linux默认。16TB文件/1EB卷。日志保证一致性。extents减少元数据。")
    _add_leaf(tree, vfs.node_id, "inode_v4", "Inode",
        "存储文件元数据。不存文件名。硬链接计数管理生命周期。")

    io_sys = _add_node(tree, os_node.node_id, "io_sys_v4", "I/O 管理",
        "I/O管理协调外设与CPU的数据交换，涉及缓冲、DMA和中断。")
    _add_leaf(tree, io_sys.node_id, "io_hierarchy_v4", "I/O 层次",
        "应用→库→系统调用→VFS→块设备→驱动→硬件。每层抽象隐藏下层细节。")
    _add_leaf(tree, io_sys.node_id, "dma_v4", "DMA",
        "设备直接读写内存，不占CPU。传输完成中断通知。高速设备必需。")

    # ═══════════════════════════════════════════════════════════
    # 6. 计算机网络 (L3)
    # ═══════════════════════════════════════════════════════════
    net = _add_node(tree, cs.node_id, "net_v4", "计算机网络",
        "计算机网络研究数据在不同设备间传输的协议和技术。核心是TCP/IP协议栈，"
        "涵盖传输层(TCP/UDP)、网络层(IP/路由)和应用层(HTTP/DNS)等层次。")

    tcpip = _add_node(tree, net.node_id, "tcpip_v4", "TCP/IP 协议栈",
        "TCP/IP四层模型：网络接口→网际→传输→应用。是互联网的通信基础。")
    transport = _add_node(tree, tcpip.node_id, "transport_v4", "传输层",
        "传输层提供端到端通信，主要有TCP(可靠)和UDP(不可靠)两种协议。")
    tcp = _add_node(tree, transport.node_id, "tcp_v4", "TCP 协议",
        "TCP(传输控制协议)提供面向连接的可靠字节流服务。头部20字节。")
    tcp_reliable = _add_node(tree, tcp.node_id, "tcp_reliable_v4", "TCP 可靠传输",
        "TCP通过序列号、确认应答、重传超时和滑动窗口实现可靠传输。")
    _add_leaf(tree, tcp_reliable.node_id, "tcp_flow_ctrl_v4", "流量控制",
        "接收窗口 rwnd 告知发送方剩余缓冲区。防止快发慢收。")
    _add_leaf(tree, tcp_reliable.node_id, "tcp_sliding_window_v4", "滑动窗口",
        "窗口内已发送等待ACK。收到ACK前移。累计确认。")
    tcp_congestion = _add_node(tree, tcp.node_id, "tcp_congestion_v4", "TCP 拥塞控制",
        "拥塞控制防止网络过载，包括慢启动、拥塞避免和快速恢复。")
    _add_leaf(tree, tcp_congestion.node_id, "tcp_slow_start_v4", "慢启动",
        "cwnd从1MSS开始，每收到ACK翻倍(指数增长)。到达ssthresh转拥塞避免。")
    _add_leaf(tree, tcp_congestion.node_id, "tcp_congestion_avoid_v4", "拥塞避免",
        "cwnd≥ssthresh后每RTT加1MSS(线性增长)。AIMD原则。")
    _add_leaf(tree, tcp_congestion.node_id, "tcp_bbr_v4", "TCP BBR",
        "Google提出。基于带宽和RTT建模，不依赖丢包。高吞吐低延迟。")
    tcp_state = _add_node(tree, tcp.node_id, "tcp_state_v4", "TCP 状态机",
        "TCP连接经历11种状态，通过三次握手建立、四次挥手关闭。")
    _add_leaf(tree, tcp_state.node_id, "tcp_state_trans_v4", "状态转移详解",
        "11种状态：CLOSED/LISTEN/SYN_SENT/SYN_RCVD/ESTABLISHED等。各状态间通过包触发转移。")
    _add_leaf(tree, tcp_state.node_id, "tcp_timewait_v4", "TIME_WAIT 详解",
        "主动关闭方发送最后ACK后进入TIME_WAIT约60秒。确保ACK到达且旧包消失。")
    tcp_handshake = _add_node(tree, tcp.node_id, "tcp_handshake_v4", "TCP 连接管理",
        "三次握手建立连接，四次挥手关闭连接。")
    _add_leaf(tree, tcp_handshake.node_id, "three_way_v4", "三次握手",
        "SYN→SYN+ACK→ACK。确认收发能力。同步初始序列号。")
    _add_leaf(tree, tcp_handshake.node_id, "four_way_v4", "四次挥手",
        "FIN→ACK→FIN→ACK。全双工关闭(需4次)。TIME_WAIT等待2MSL。")

    udp = _add_node(tree, transport.node_id, "udp_v4", "UDP 协议",
        "UDP无连接不可靠，头部仅8字节。适合实时应用和简单请求。")
    _add_leaf(tree, udp.node_id, "udp_basics_v4", "UDP 基础",
        "无连接、不可靠、无拥塞。8字节头。支持广播多播。DNS/QUIC/视频流。")
    _add_leaf(tree, udp.node_id, "udp_vs_tcp_v4", "UDP vs TCP",
        "UDP快但丢包不重传。TCP可靠但延迟高。TCP 20字节头 vs UDP 8字节。")

    network_layer = _add_node(tree, tcpip.node_id, "network_layer_v4", "网络层",
        "网络层负责包的路由和转发，核心协议是IP协议。")
    ip_proto = _add_node(tree, network_layer.node_id, "ip_proto_v4", "IP 协议",
        "IP协议提供无连接的数据报传输服务。IPv4(32位)/IPv6(128位)。")
    _add_leaf(tree, ip_proto.node_id, "ipv4_v4", "IPv4",
        "32位地址点分十进制。A/B/C/D/E类。NAT缓解地址枯竭。")
    _add_leaf(tree, ip_proto.node_id, "ipv6_v4", "IPv6",
        "128位地址冒分十六进制。无NAT无校验和。即插即用SLAAC。")
    _add_leaf(tree, ip_proto.node_id, "nat_v4", "NAT",
        "私网↔公网映射。SNAT内网上网/DNAT端口映射。破坏端到端透明。")

    routing = _add_node(tree, network_layer.node_id, "routing_v4", "路由协议",
        "路由协议决定数据包从源到目的的最佳路径。")
    _add_leaf(tree, routing.node_id, "ospf_v4", "OSPF",
        "链路状态路由。Dijkstra SPF计算路径。区域分层。收敛快。")
    _add_leaf(tree, routing.node_id, "bgp_v4", "BGP",
        "互联网的粘合剂。路径向量协议。iBGP(AS内)/eBGP(AS间)。")

    app_layer = _add_node(tree, tcpip.node_id, "app_layer_v4", "应用层协议",
        "应用层协议为应用程序提供网络服务，包括HTTP、DNS、FTP等。")
    http = _add_node(tree, app_layer.node_id, "http_v4", "HTTP",
        "HTTP是Web的基石，请求-响应模型，无状态。HTTP/1.1文本协议。")
    http_core = _add_node(tree, http.node_id, "http_core_v4", "HTTP 核心",
        "HTTP核心包括请求方法、状态码和缓存机制。")
    _add_leaf(tree, http_core.node_id, "http_methods_v4", "HTTP 方法",
        "GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS。")
    _add_leaf(tree, http_core.node_id, "http_status_v4", "HTTP 状态码",
        "2xx成功/3xx重定向/4xx客户端错误/5xx服务端错误。")
    _add_leaf(tree, http_core.node_id, "https_v4", "HTTPS",
        "HTTP+SSL/TLS。TLS握手加密+证书验证。TLS 1.3只需1个RTT。")
    http2 = _add_node(tree, http.node_id, "http2_v4", "HTTP/2",
        "HTTP/2二进制分帧，多路复用。头部压缩HPACK。服务器推送。")
    _add_leaf(tree, http2.node_id, "http2_multiplex_v4", "多路复用",
        "单一TCP连接并行传输多个流。解决HTTP/1.1队头阻塞。")
    http3 = _add_node(tree, http.node_id, "http3_v4", "HTTP/3",
        "HTTP/3基于QUIC(UDP)，彻底解决TCP队头阻塞。0-RTT重连。")
    _add_leaf(tree, http3.node_id, "quic_v4", "QUIC 协议",
        "Google设计，IETF标准化。基于UDP+内置TLS1.3。连接迁移。")
    _add_leaf(tree, app_layer.node_id, "dns_v4", "DNS",
        "域名→IP。递归/迭代查询。A/AAAA/CNAME/MX记录。根服务器13组。")
    _add_leaf(tree, app_layer.node_id, "cdn_v4", "CDN",
        "全球缓存节点。请求到最近节点。回源获取。边缘计算。")

    # ═══════════════════════════════════════════════════════════
    # 7. 数据库 (L3)
    # ═══════════════════════════════════════════════════════════
    db = _add_node(tree, cs.node_id, "db_v4", "数据库",
        "数据库系统化地存储和管理数据。关系型数据库(SQL)和非关系型数据库(NoSQL)是最常用的两类。"
        "核心概念包括SQL、索引、事务、ACID和分布式。")

    rdb = _add_node(tree, db.node_id, "rdb_v4", "关系型数据库",
        "关系型数据库以表(关系)组织数据，支持ACID事务和SQL查询。")
    sql = _add_node(tree, rdb.node_id, "sql_v4", "SQL 语言",
        "SQL是关系数据库的标准查询语言，分DDL/DML/DCL。")
    _add_leaf(tree, sql.node_id, "ddl_v4", "DDL 数据定义",
        "CREATE/ALTER/DROP TABLE。数据类型和约束。PRIMARY KEY/FOREIGN KEY。")
    _add_leaf(tree, sql.node_id, "dml_v4", "DML 数据操作",
        "SELECT/INSERT/UPDATE/DELETE。WHERE/ORDER BY/LIMIT。")
    _add_leaf(tree, sql.node_id, "join_v4", "JOIN 连接",
        "INNER/LEFT/RIGHT/FULL OUTER/CROSS JOIN。")

    index_db = _add_node(tree, rdb.node_id, "index_db_v4", "索引",
        "索引是加速数据查询的辅助数据结构，核心类型包括B+树和哈希索引。")
    index_types = _add_node(tree, index_db.node_id, "index_types_v4", "索引类型",
        "不同索引类型适用于不同查询模式。")
    _add_leaf(tree, index_types.node_id, "btree_index_v4", "B+ 树索引",
        "所有数据在叶子节点，内部只有键值指针。叶子链表。InnoDB聚簇索引。")
    _add_leaf(tree, index_types.node_id, "hash_index_v4", "哈希索引",
        "键哈希映射。仅等值查询。Memery引擎默认。O(1)查询。")
    _add_leaf(tree, index_types.node_id, "fulltext_index_v4", "全文索引",
        "倒排索引：词→文档列表。MATCH AGAINST。")

    tx = _add_node(tree, rdb.node_id, "tx_v4", "事务",
        "事务是数据库操作的基本单元，具有ACID特性。")
    tx_ac = _add_node(tree, tx.node_id, "tx_ac_v4", "ACID 特性",
        "ACID保证事务的正确执行。")
    _add_leaf(tree, tx_ac.node_id, "atomicity_v4", "原子性",
        "事务全部成功或全部失败。WAL实现原子性。")
    _add_leaf(tree, tx_ac.node_id, "isolation_v4", "隔离性",
        "并发事务互不干扰。MVCC实现隔离。")
    _add_leaf(tree, tx_ac.node_id, "durability_v4", "持久性",
        "已提交结果永久保存。Redo log重放。")

    isolation = _add_node(tree, tx.node_id, "isolation_v4", "隔离级别",
        "四种隔离级别控制并发事务的可见性。")
    _add_leaf(tree, isolation.node_id, "read_uncommitted_v4", "读未提交",
        "最低级。可读到未提交数据(脏读)。几乎不用。")
    _add_leaf(tree, isolation.node_id, "read_committed_v4", "读已提交",
        "只能读已提交。避免脏读。不可重复读。")
    _add_leaf(tree, isolation.node_id, "repeatable_read_v4", "可重复读",
        "多次读同记录结果一致。InnoDB默认。MVCC快照读。")
    _add_leaf(tree, isolation.node_id, "serializable_v4", "可串行化",
        "最高级。事务串行执行。无并发问题。性能最低。")

    nosql = _add_node(tree, db.node_id, "nosql_v4", "NoSQL",
        "NoSQL数据库提供灵活的数据模型和水平扩展能力。")
    _add_leaf(tree, nosql.node_id, "redis_v4", "Redis",
        "内存键值数据库。String/List/Set/SortedSet/Hash/Bitmap/GEO。")
    _add_leaf(tree, nosql.node_id, "mongodb_v4", "MongoDB",
        "文档数据库(BSON)。无Schema。副本集+分片水平扩展。")

    dist_db = _add_node(tree, db.node_id, "dist_db_v4", "分布式数据库",
        "分布式数据库将数据分布在多台机器上。CAP定理是理论基础。")
    _add_leaf(tree, dist_db.node_id, "cap_v4", "CAP 定理",
        "Consistency/Availability/Partition tolerance 三者不可兼得。")
    _add_leaf(tree, dist_db.node_id, "raft_v4", "Raft 共识算法",
        "领导者选举→日志复制→安全性。Etcd/Consul/TiKV使用。")
    _add_leaf(tree, dist_db.node_id, "sharding_v4", "分片",
        "水平拆分：按哈希/范围/列表分布。分片键选择关键。")

    # ═══════════════════════════════════════════════════════════
    # 8. Linux (L3)
    # ═══════════════════════════════════════════════════════════
    linux = _add_node(tree, cs.node_id, "linux_v4", "Linux",
        "Linux是一种自由开源的操作系统内核，广泛应用于服务器、嵌入式系统和云计算。"
        "以其稳定性、安全性和灵活性著称，是互联网基础设施的核心。")

    linux_admin = _add_node(tree, linux.node_id, "linux_admin_v4", "系统管理",
        "Linux系统管理包括文件操作、文本处理、进程管理和权限控制等日常运维技能。")
    linux_fs = _add_node(tree, linux_admin.node_id, "linux_fs_v4", "文件管理",
        "Linux文件管理命令包括ls/cd/cp/mv/rm/find等核心工具。")
    _add_leaf(tree, linux_fs.node_id, "ls_cd_pwd_v4", "ls/cd/pwd",
        "ls -l详细 -a隐藏 -h可读。cd ~回家。pwd当前路径。")
    _add_leaf(tree, linux_fs.node_id, "cp_mv_rm_v4", "cp/mv/rm",
        "cp -r递归目录。mv移动重命名。rm -rf慎用。")
    _add_leaf(tree, linux_fs.node_id, "find_locate_v4", "find/locate",
        "find按名/型/大查找。locate快但非实时。")
    _add_leaf(tree, linux_fs.node_id, "tar_gzip_v4", "tar/gzip",
        "tar -czf打包压缩。tar -xzf解压。gz/bz2/xz。")

    linux_text = _add_node(tree, linux_admin.node_id, "linux_text_v4", "文本处理",
        "Linux文本处理三剑客：grep/sed/awk。")
    _add_leaf(tree, linux_text.node_id, "grep_v4", "grep",
        "grep模式搜索。常用：-i忽略 -r递归 -n行号 -v反向。")
    _add_leaf(tree, linux_text.node_id, "sed_v4", "sed",
        "sed 's/old/new/g'替换。sed -n '1,10p'打印范围。")
    _add_leaf(tree, linux_text.node_id, "awk_v4", "awk",
        "awk '{print $1,$NF}'列处理。条件统计和报表。")
    _add_leaf(tree, linux_text.node_id, "vim_basics_v4", "Vim",
        "三种模式。:wq保存 :q!不存。dd/yy/p。")

    linux_proc = _add_node(tree, linux_admin.node_id, "linux_proc_v4", "进程管理",
        "Linux进程管理涉及ps/top查看、kill控制和systemd服务管理。")
    _add_leaf(tree, linux_proc.node_id, "ps_top_v4", "ps/top",
        "ps aux所有进程。top实时视图。htop增强交互。")
    _add_leaf(tree, linux_proc.node_id, "systemd_v4", "systemd",
        "系统和服务管理器。systemctl/unit/journalctl。")
    _add_leaf(tree, linux_proc.node_id, "kill_nice_v4", "kill/nice",
        "kill -9强制。nice/renice调优先级。")

    linux_perm = _add_node(tree, linux_admin.node_id, "linux_perm_v4", "权限管理",
        "Linux权限三位一组(rwx)。chmod/chown管理。sudo授权。")
    _add_leaf(tree, linux_perm.node_id, "chmod_chown_v4", "chmod/chown",
        "chmod 755(rwxr-xr-x)。chown user:group。SUID/SGID/sticky。")
    _add_leaf(tree, linux_perm.node_id, "sudo_v4", "sudo",
        "普通用户以root执行。visudo编辑配置。NOPASSWD免密。")

    linux_net = _add_node(tree, linux_admin.node_id, "linux_net_v4", "网络配置",
        "Linux网络配置包括ip/ss命令、iptables防火墙和SSH远程连接。")
    _add_leaf(tree, linux_net.node_id, "ip_ss_v4", "ip/ss",
        "ip addr/link/route。ss -tuln 替代netstat。")
    _add_leaf(tree, linux_net.node_id, "iptables_v4", "iptables/nftables",
        "filter表INPUT/FORWARD/OUTPUT。规则匹配动作。")
    _add_leaf(tree, linux_net.node_id, "ssh_v4", "SSH",
        "ssh user@host。密钥认证。scp复制。端口转发。")

    shell = _add_node(tree, linux.node_id, "shell_v4", "Shell 脚本",
        "Bash脚本是Linux自动化的核心工具，支持变量、控制流、管道和重定向。")
    _add_leaf(tree, shell.node_id, "bash_basics_v4", "Bash 基础",
        "#!/bin/bash。变量${var}。$0/$1/$#/$?。$(cmd)替换。")
    _add_leaf(tree, shell.node_id, "bash_control_v4", "Bash 控制流",
        "if/for/while/case。条件判断[[ ]]。")
    _add_leaf(tree, shell.node_id, "bash_pipe_v4", "管道与重定向",
        "| 管道。> < 重定向。2>&1 合并错误。tee分流。")

    # ═══════════════════════════════════════════════════════════
    # 9. 软件工程 (L3)
    # ═══════════════════════════════════════════════════════════
    se = _add_node(tree, cs.node_id, "se_v4", "软件工程",
        "软件工程是用系统化方法指导软件开发、运行和维护的工程学科。"
        "涵盖设计模式、版本控制、CI/CD、架构模式和项目管理等。")

    design_patterns = _add_node(tree, se.node_id, "design_patterns_v4", "设计模式",
        "设计模式是软件设计中常见问题的可复用解决方案。分为创建型、结构型和行为型三类。")
    creational = _add_node(tree, design_patterns.node_id, "creational_v4", "创建型模式",
        "创建型模式抽象对象实例化过程，使系统独立于对象的创建方式。")
    _add_leaf(tree, creational.node_id, "singleton_v4", "单例模式",
        "保证类只有一个实例。饿汉/懒汉。Spring Bean默认单例。")
    _add_leaf(tree, creational.node_id, "factory_v4", "工厂模式",
        "简单工厂/工厂方法/抽象工厂。解耦客户端和具体类。")
    _add_leaf(tree, creational.node_id, "builder_v4", "建造者模式",
        "分步构造复杂对象。链式调用。StringBuilder/Lombok@Builder。")
    structural = _add_node(tree, design_patterns.node_id, "structural_v4", "结构型模式",
        "结构型模式关注类和对象的组合。")
    _add_leaf(tree, structural.node_id, "adapter_v4", "适配器模式",
        "将不兼容接口转为客户端期望的接口。类适配器/对象适配器。")
    _add_leaf(tree, structural.node_id, "proxy_v4", "代理模式",
        "为对象提供替身控制访问。远程/虚拟/保护代理。AOP底层。")
    _add_leaf(tree, structural.node_id, "decorator_v4", "装饰器模式",
        "动态给对象加职责。比继承灵活。IO流BufferedInputStream。")
    behavioral = _add_node(tree, design_patterns.node_id, "behavioral_v4", "行为型模式",
        "行为型模式关注对象间的交互和职责分配。")
    _add_leaf(tree, behavioral.node_id, "observer_v4", "观察者模式",
        "一对多依赖，状态变化通知所有观察者。发布-订阅。")
    _add_leaf(tree, behavioral.node_id, "strategy_v4", "策略模式",
        "定义可互相替换的算法族。消除if-else。")
    _add_leaf(tree, behavioral.node_id, "chain_v4", "责任链模式",
        "多个处理者串联。请求沿链传递直到被处理。Filter/Interceptor。")

    vcs = _add_node(tree, se.node_id, "vcs_v4", "版本控制",
        "版本控制记录文件变更历史，支持协作开发和回滚。Git是当前最流行的版本控制系统。")
    git = _add_node(tree, vcs.node_id, "git_v4", "Git",
        "Git是分布式版本控制系统，核心概念包括提交、分支、合并和远程。")
    _add_leaf(tree, git.node_id, "git_basics_v4", "Git 基础",
        "init/clone/add/commit/push/pull。暂存区工作流。")
    _add_leaf(tree, git.node_id, "git_branch_v4", "分支与合并",
        "branch/checkout/merge。冲突解决。rebase变基。Git Flow。")

    cicd = _add_node(tree, se.node_id, "cicd_v4", "CI/CD",
        "持续集成(CI)和持续部署(CD)实现代码自动构建、测试和部署。")
    _add_leaf(tree, cicd.node_id, "ci_basics_v4", "持续集成 CI",
        "每次推送自动构建测试。Jenkins/GitLab CI/GitHub Actions。")
    _add_leaf(tree, cicd.node_id, "cd_basics_v4", "持续部署 CD",
        "CI通过自动部署到环境。蓝绿/滚动/金丝雀发布。")

    arch = _add_node(tree, se.node_id, "arch_v4", "架构模式",
        "架构模式是软件系统的顶层设计蓝图。")
    _add_leaf(tree, arch.node_id, "microservices_v4", "微服务",
        "独立服务独立部署。服务间轻量API通信。DB per Service。")
    _add_leaf(tree, arch.node_id, "event_driven_v4", "事件驱动架构",
        "组件通过事件通信。松耦合高扩展。Kafka/RabbitMQ。")

    # ═══════════════════════════════════════════════════════════
    # 10. 机器学习 (L3)
    # ═══════════════════════════════════════════════════════════
    ml = _add_node(tree, cs.node_id, "ml_v4", "机器学习",
        "机器学习让计算机从数据中学习规律，无需显式编程。"
        "分为监督学习、无监督学习、半监督学习和强化学习。深度学习是其重要分支。")

    supervised = _add_node(tree, ml.node_id, "supervised_v4", "监督学习",
        "监督学习使用有标签的训练数据学习输入到输出的映射。")
    regression = _add_node(tree, supervised.node_id, "regression_v4", "回归",
        "回归预测连续值。线性回归和逻辑回归是最基础的回归算法。")
    _add_leaf(tree, regression.node_id, "linear_reg_v4", "线性回归",
        "y=wx+b。最小二乘法MSE。梯度下降优化。L1/L2正则化。")
    _add_leaf(tree, regression.node_id, "logistic_reg_v4", "逻辑回归",
        "sigmoid输出概率。交叉熵损失。二分类标准算法。")
    classification = _add_node(tree, supervised.node_id, "classification_v4", "分类",
        "分类预测离散类别标签。")
    _add_leaf(tree, classification.node_id, "decision_tree_v4", "决策树",
        "树状决策。信息增益/基尼系数分裂。可解释性强。")
    _add_leaf(tree, classification.node_id, "svm_v4", "SVM",
        "最大间隔超平面。核技巧RBF。适合小规模高维数据。")
    _add_leaf(tree, classification.node_id, "knn_v4", "KNN",
        "K个最近邻投票。距离度量。无训练过程。")
    ensemble = _add_node(tree, supervised.node_id, "ensemble_v4", "集成学习",
        "集成学习组合多个弱学习器提升效果。")
    _add_leaf(tree, ensemble.node_id, "random_forest_v4", "随机森林",
        "Bagging+特征随机。多树投票。不易过拟合。")
    _add_leaf(tree, ensemble.node_id, "gbdt_v4", "梯度提升树 GBDT",
        "逐步添加残差拟合。XGBoost/LightGBM。表格数据最佳。")

    unsupervised = _add_node(tree, ml.node_id, "unsupervised_v4", "无监督学习",
        "无监督学习从无标签数据中发现结构。")
    _add_leaf(tree, unsupervised.node_id, "kmeans_v4", "K-Means",
        "迭代聚类。K值需预设。肘部法确定K。")
    _add_leaf(tree, unsupervised.node_id, "pca_v4", "PCA",
        "主成分分析降维。去相关去噪。最大化保留方差。")

    dl = _add_node(tree, ml.node_id, "dl_v4", "深度学习",
        "深度学习使用多层神经网络学习复杂模式和表示。核心架构包括CNN、RNN和Transformer。")
    nn_basics = _add_node(tree, dl.node_id, "nn_basics_v4", "神经网络基础",
        "神经网络由神经元层叠而成，通过反向传播学习。")
    _add_leaf(tree, nn_basics.node_id, "perceptron_v4", "感知机",
        "y=σ(w·x+b)。多层感知机(MLP)可拟合任意函数。")
    _add_leaf(tree, nn_basics.node_id, "activation_v4", "激活函数",
        "ReLU(max(0,x))最常用。Sigmoid/Tanh/Softmax。")
    _add_leaf(tree, nn_basics.node_id, "backpropagation_v4", "反向传播",
        "链式法则计算梯度。前向→损失→反向→更新。")
    _add_leaf(tree, nn_basics.node_id, "optimizers_v4", "优化器",
        "SGD/Adam/RMSprop/AdamW。学习率调度。")
    cnn = _add_node(tree, dl.node_id, "cnn_v4", "CNN",
        "卷积神经网络通过卷积核提取局部特征，适合图像处理。")
    _add_leaf(tree, cnn.node_id, "conv_layer_v4", "卷积层",
        "卷积核滑动提取特征。参数共享。padding/stride。")
    _add_leaf(tree, cnn.node_id, "pooling_v4", "池化层",
        "最大池化/平均池化。降采样。平移不变性。")
    _add_leaf(tree, cnn.node_id, "cnn_arch_v4", "经典 CNN 架构",
        "LeNet/AlexNet/VGG/ResNet/Inception。")
    transformer = _add_node(tree, dl.node_id, "transformer_v4", "Transformer",
        "Transformer 利用自注意力机制处理序列，是BERT/GPT等大模型的基础架构。")
    _add_leaf(tree, transformer.node_id, "transformer_arch_v4", "Transformer 架构",
        "编码器+解码器。自注意力+前馈网络。位置编码。LayerNorm。")
    _add_leaf(tree, transformer.node_id, "self_attention_v4", "自注意力",
        "QKV机制：softmax(QK^T/√d)V。O(n²)复杂度。")
    _add_leaf(tree, transformer.node_id, "pretrain_finetune_v4", "预训练+微调",
        "大规模预训练+下游微调。BERT MLM/GPT自回归/LoRA。")

    nlp = _add_node(tree, ml.node_id, "nlp_v4", "自然语言处理",
        "NLP让计算机理解、生成和处理人类语言。")
    _add_leaf(tree, nlp.node_id, "tokenization_v4", "分词",
        "BPE/WordPiece/Unigram。SentencePiece。")
    _add_leaf(tree, nlp.node_id, "embedding_v4", "词嵌入",
        "Word2Vec/GloVe/FastText。BERT上下文嵌入超越静态向量。")
    _add_leaf(tree, nlp.node_id, "llm_v4", "大语言模型 LLM",
        "GPT/Claude/LLaMA/Qwen。Scaling Law。RLHF/DPO对齐。")

    # ═══════════════════════════════════════════════════════════
    # 11. 前端开发 (L3)
    # ═══════════════════════════════════════════════════════════
    fe = _add_node(tree, cs.node_id, "fe_v4", "前端开发",
        "前端开发构建用户界面和交互体验。核心技术HTML/CSS/JavaScript，现代框架React/Vue/Angular。")
    _add_leaf(tree, fe.node_id, "html_basics_v4", "HTML",
        "标签结构<html><head><body>。语义化header/nav/main/section/article/footer。")
    _add_leaf(tree, fe.node_id, "css_box_model_v4", "CSS 盒模型",
        "content/padding/border/margin。box-sizing。display。")
    _add_leaf(tree, fe.node_id, "css_layout_v4", "CSS 布局",
        "Flexbox一维/Grid二维。媒体查询响应式。")
    _add_leaf(tree, fe.node_id, "react_hooks_v4", "React Hooks",
        "useState/useEffect/useRef/useContext/useMemo/useCallback。")
    _add_leaf(tree, fe.node_id, "react_vdom_v4", "虚拟 DOM",
        "JSX编译为虚拟DOM。diff算法。Fiber可中断渲染。")
    _add_leaf(tree, fe.node_id, "nextjs_v4", "Next.js",
        "React全栈框架。SSR/SSG/ISR。App Router。Server Components。")
    _add_leaf(tree, fe.node_id, "vite_v4", "Vite",
        "基于ESM的构建工具。毫秒HMR。Rollup打包。取代Webpack趋势。")

    # ═══════════════════════════════════════════════════════════
    # 12. 后端开发 (L3)
    # ═══════════════════════════════════════════════════════════
    be = _add_node(tree, cs.node_id, "be_v4", "后端开发",
        "后端开发处理服务器端逻辑、API设计和数据存储。关键技术包括REST、gRPC、GraphQL和认证授权。")
    _add_leaf(tree, be.node_id, "rest_api_v4", "REST API",
        "资源导向。GET/POST/PUT/DELETE。状态无关。OpenAPI文档。")
    _add_leaf(tree, be.node_id, "grpc_v4", "gRPC",
        "高性能RPC。Protobuf序列化。HTTP/2。流式通信。")
    _add_leaf(tree, be.node_id, "graphql_v4", "GraphQL",
        "客户端声明字段。单一端点。避免过/欠获取。")
    _add_leaf(tree, be.node_id, "auth_v4", "认证与授权",
        "JWT无状态/OAuth2.0授权/Session-Cookie有状态/RBAC权限。")

    # ═══════════════════════════════════════════════════════════
    # 13. 系统设计 (L3)
    # ═══════════════════════════════════════════════════════════
    sd = _add_node(tree, cs.node_id, "sd_v4", "系统设计",
        "系统设计构建大规模分布式系统的技术和架构，涉及分布式系统、缓存、消息队列等。")
    dist_sys = _add_node(tree, sd.node_id, "dist_sys_v4", "分布式系统",
        "分布式系统由多台计算机协同工作，对外表现为单一系统。挑战包括一致性、分区容错。")
    _add_leaf(tree, dist_sys.node_id, "consistency_models_v4", "一致性模型",
        "强一致性/最终一致性。Quorum读写。PACELC定理。")
    _add_leaf(tree, dist_sys.node_id, "distributed_id_v4", "分布式 ID",
        "雪花算法(Snowflake)：时间戳+机器ID+序列号。UUID。")
    cache = _add_node(tree, sd.node_id, "cache_v4", "缓存",
        "缓存提升系统性能。核心策略包括Cache Aside和分布式缓存。")
    _add_leaf(tree, cache.node_id, "cache_strategies_v4", "缓存策略",
        "Cache Aside旁路。穿透/雪崩/击穿。布隆过滤器/互斥锁。")
    _add_leaf(tree, cache.node_id, "distributed_cache_v4", "分布式缓存",
        "Redis Cluster/一致性哈希/虚拟节点。多级缓存L1+L2。")
    mq = _add_node(tree, sd.node_id, "mq_v4", "消息队列",
        "消息队列解耦服务，提供异步通信、削峰填谷和流量控制。")
    _add_leaf(tree, mq.node_id, "kafka_v4", "Kafka",
        "Topic/Partition/Offset。高吞吐顺序IO。发布-订阅。")
    _add_leaf(tree, mq.node_id, "rabbitmq_v4", "RabbitMQ",
        "AMQP。Exchange(Topic/Direct/Fanout)。消息确认ACK。")

    # ═══════════════════════════════════════════════════════════
    # 14. 网络安全 (L3)
    # ═══════════════════════════════════════════════════════════
    sec = _add_node(tree, cs.node_id, "sec_v4", "网络安全",
        "网络安全保护系统、网络和数据免受攻击。核心领域包括密码学、Web安全和网络攻防。")
    crypto = _add_node(tree, sec.node_id, "crypto_v4", "密码学",
        "密码学通过加密保护信息安全。对称和非对称加密是两大体系。")
    _add_leaf(tree, crypto.node_id, "symmetric_enc_v4", "对称加密",
        "AES/Rijndael。GCM模式推荐。密钥分发问题。")
    _add_leaf(tree, crypto.node_id, "asymmetric_enc_v4", "非对称加密",
        "RSA/ECC。公钥加密私钥解密。数字签名。")
    _add_leaf(tree, crypto.node_id, "hash_v4", "哈希函数",
        "SHA-256/3。不可逆抗碰撞。bcrypt/argon2密码哈希。")
    web_sec = _add_node(tree, sec.node_id, "web_sec_v4", "Web 安全",
        "Web安全防御常见攻击。")
    _add_leaf(tree, web_sec.node_id, "xss_v4", "XSS 跨站脚本",
        "反射/存储/DOM型。CSP/HttpOnly Cookie防护。")
    _add_leaf(tree, web_sec.node_id, "csrf_v4", "CSRF",
        "用户已登录被诱导操作。CSRF Token/SameSite Cookie防护。")
    _add_leaf(tree, web_sec.node_id, "sql_injection_v4", "SQL注入",
        "恶意SQL拼接到查询。参数化查询/ORM防护。")

    # ═══════════════════════════════════════════════════════════
    # 15. DevOps (L3)
    # ═══════════════════════════════════════════════════════════
    devops = _add_node(tree, cs.node_id, "devops_v4", "DevOps",
        "DevOps通过自动化工具和文化实践，实现开发和运维的协作。核心包括容器化、Kubernetes和可观测性。")

    container = _add_node(tree, devops.node_id, "container_v4", "容器化",
        "容器提供轻量级的应用隔离和打包方式。Docker是容器化的标准。")
    docker = _add_node(tree, container.node_id, "docker_v4", "Docker",
        "Docker容器化平台。镜像分层UnionFS。容器隔离namespace+cgroups。")
    _add_leaf(tree, docker.node_id, "docker_image_v4", "Docker 镜像",
        "只读模板分层构建。Dockerfile定义步骤。镜象层缓存加速。")
    _add_leaf(tree, docker.node_id, "docker_container_v4", "Docker 容器",
        "镜像运行实例。run/exec/stop/rm。端口映射-p。卷挂载-v。")
    _add_leaf(tree, docker.node_id, "docker_network_v4", "Docker 网络",
        "bridge默认/host共享/overlay跨主机。docker-compose多容器。")
    k8s = _add_node(tree, container.node_id, "k8s_v4", "Kubernetes",
        "K8s容器编排平台。管理容器化应用的部署、扩展和运维。")
    _add_leaf(tree, k8s.node_id, "k8s_pod_v4", "Pod",
        "最小调度单元。一个或多个容器共享网络/存储。Sidecar模式。")
    _add_leaf(tree, k8s.node_id, "k8s_deploy_v4", "Deployment",
        "声明式Pod更新。滚动更新/回滚。HPA水平扩缩。")
    _add_leaf(tree, k8s.node_id, "k8s_service_v4", "Service",
        "Pod稳定访问入口。ClusterIP/NodePort/LoadBalancer/Ingress。")

    monitoring = _add_node(tree, devops.node_id, "monitoring_v4", "可观测性",
        "可观测性包括监控(Metrics)、追踪(Trace)和日志(Log)三大支柱。")
    _add_leaf(tree, monitoring.node_id, "prometheus_v4", "Prometheus",
        "时序数据库+监控。Pull模式抓取。PromQL查询。Alertmanager告警。")
    _add_leaf(tree, monitoring.node_id, "grafana_v4", "Grafana",
        "可视化仪表盘。多数据源。告警规则。面板类型丰富。")

    cloud = _add_node(tree, devops.node_id, "cloud_v4", "云原生基础",
        "云原生技术利用云计算模型构建和运行应用。")
    _add_leaf(tree, cloud.node_id, "aws_basics_v4", "AWS",
        "EC2/S3/RDS/Lambda/CloudFront/VPC。IAM权限管理。")
    _add_leaf(tree, cloud.node_id, "terraform_v4", "Terraform",
        "IaC(HCL语言)。plan/apply/destroy。状态文件。多provider。")

    # ═══════════════════════════════════════════════════════════
    # 跨域引用
    # ═══════════════════════════════════════════════════════════
    cross_pairs = [
        ("py_generator_v4", "py_async_await_v4"), ("py_iterator_proto_v4", "py_for_v4"),
        ("py_class_def_v4", "py_except_v4"), ("py_decorator_v4", "py_adv_deco_v4"),
        ("java_vm_basics_v4", "java_oop_v4"), ("java_map_v4", "red_black_tree_v4"),
        ("js_promise_v4", "js_event_loop_v4"), ("js_closure_v4", "js_async_await_v4"),
        ("quick_sort_v4", "merge_sort_v4"), ("quick_sort_v4", "heap_sort_v4"),
        ("binary_search_v4", "bst_search_v4"), ("bfs_v4", "dfs_v4"),
        ("dp_basics_v4", "knapsack_v4"), ("dijkstra_v4", "floyd_v4"),
        ("avl_tree_v4", "red_black_tree_v4"), ("union_find_v4", "kruskal_v4"),
        ("fcfs_v4", "rr_v4"), ("mutex_lock_v4", "semaphore_v4"),
        ("paging_v4", "page_replacement_v4"), ("mutex_lock_v4", "deadlock_v4"),
        ("three_way_v4", "four_way_v4"), ("tcp_slow_start_v4", "tcp_congestion_avoid_v4"),
        ("http_methods_v4", "http_status_v4"), ("http2_multiplex_v4", "quic_v4"),
        ("btree_index_v4", "hash_index_v4"), ("atomicity_v4", "isolation_v4"),
        ("cap_v4", "raft_v4"), ("b_tree_v4", "btree_index_v4"),
        ("decision_tree_v4", "random_forest_v4"), ("transformer_arch_v4", "self_attention_v4"),
        ("docker_image_v4", "docker_container_v4"), ("k8s_pod_v4", "k8s_deploy_v4"),
        ("prometheus_v4", "grafana_v4"),
        ("linear_reg_v4", "logistic_reg_v4"), ("xss_v4", "csrf_v4"),
        ("singleton_v4", "proxy_v4"), ("react_hooks_v4", "react_vdom_v4"),
        ("supervised_v4", "unsupervised_v4"), ("nn_basics_v4", "cnn_v4"),
    ]
    for a, b in cross_pairs:
        tree.add_cross_reference(a, b)

    # 树已建成，节点自带语义向量，无需 pool_all
    # pool_all 会稀释枝干自身的向量，所以不调用

    print(f"\n🌳 v4 知识树构建完成:")
    stats = tree.stats()
    print(f"  总节点: {stats['total_nodes']}  |  叶子: {stats['leaf_nodes']}")
    print(f"  深度: {stats['depth']} 层  |  跨域引用: {len(cross_pairs)}")
    return tree


build_v3 = build_v4
build_v2 = build_v4
