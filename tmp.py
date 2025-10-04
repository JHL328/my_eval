from math_verify import parse, verify

# 测试基本功能
def test_basic():
    # 测试集合运算
    gold = parse("${1,3} \\cup {2,4}$")
    answer = parse("${1,2,3,4}$")
    result = verify(gold, answer)
    print(f"集合运算测试: {result}")  # 应该是 True
    
    # 测试数值比较
    gold2 = parse("$\\sqrt{2}$")
    answer2 = parse("1.4142135623730951")
    result2 = verify(gold2, answer2)
    print(f"数值比较测试: {result2}")  # 应该是 True
    
    # 测试表达式等价
    gold3 = parse("$x^2 + 2x + 1$")
    answer3 = parse("$(x+1)^2$")
    result3 = verify(gold3, answer3)
    print(f"表达式等价测试: {result3}")  # 应该是 True

if __name__ == "__main__":
    test_basic()