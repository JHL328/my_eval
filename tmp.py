from math_verify import parse, verify

# 测试基本功能
def test_basic():
    # 测试集合运算
    gold = parse("${1,3} \\cup {2,4}$")
    answer = parse("${1,2,3,4}$")
    result = verify(gold, answer)
    print(f"集合运算测试: {result}")  # 应该是 True
    
    # 测试数值比较
    gold2 = parse("18")
    answer2 = parse("Janet has 16 eggs per day. She eats 3 for breakfast and bakes 4 muffins. That is 3 + 4 = 7 eggs. The remainder is 16 - 7 = 9 eggs. She sells 9 eggs at $2 per fresh duck egg. 9 x 2 = $18. The answer is $18.")
    print("gold2:{}".format(gold2), type(gold2))
    print("answer2:{}".format(answer2), type(answer2))
    result2 = verify(gold2, answer2)
    print(f"数值比较测试: {result2}")  # 应该是 True
    
    # 测试表达式等价
    gold3 = parse("$x^2 + 2x + 1$")
    answer3 = parse("$(x+1)^2$")
    result3 = verify(gold3, answer3)
    print(f"表达式等价测试: {result3}")  # 应该是 True

if __name__ == "__main__":
    test_basic()