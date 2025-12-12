from exp_stanhop_fiats import train_and_predict

def choose_states():
    state_list = ["NSW", "QLD", "SA", "TAS", "VIC"]
    print("=== 可选州列表 ===")
    for i, s in enumerate(state_list, 1):
        print(f"{i}. {s}")
    print("0. 所有州")

    selection = input("请输入州编号（如 1 或 1,3 或 0 表示全部）：").strip()

    if selection == "0":
        return state_list
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            return [state_list[i - 1] for i in indices if 1 <= i <= 5]
        except:
            print("❌ 输入有误，请重新运行")
            exit()

def choose_sources():
    print("\n=== 请选择外部干预信息源（可多选）===")
    print("0. 无外部信息源（只用负荷数据）")
    print("1. 新闻（News）")
    print("2. 社交媒体（Reddit）")
    print("3. 政策文本（Policy）")
    print("输入组合，例如 1、12、23、123，或 0：")

    selection = input("请选择外部信息源：").strip()

    # 如果选择 0，则不使用外部信息源
    if selection == "0":
        return "0"

    # 校验输入是否合法（仅包含 1, 2, 3）
    if not selection or any(c not in '123' for c in selection):
        print("❌ 输入有误，请输入 0 或 1~3 组成的组合")
        exit()

    return ''.join(sorted(set(selection)))  # 去重排序如 321 -> 123

def main():
    print("=========== STANHOP-FIATS 电力预测训练接口 ===========")
    states = choose_states()
    sources = choose_sources()

    try:
        epochs = int(input("请输入训练轮数（默认 20）: ") or 20)
        batch_size = int(input("请输入 batch_size（默认 32）: ") or 32)
    except:
        print("❌ 输入无效，请输入整数")
        exit()

    device = input("请输入设备（默认 cuda，如无GPU请输入 cpu）: ") or "cuda"

    for state in states:
        train_and_predict(
            state=state,
            sources=sources,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

if __name__ == "__main__":
    main()
