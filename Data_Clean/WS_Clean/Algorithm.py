# 计算涉案金额
def Calculate_Amount(Numberlist):
    from decimal import Decimal
    Numberlist = [Decimal(str(i)) for i in Numberlist]
    Numberlist_pairs = [(index, item) for index, item in enumerate(Numberlist)]
    Number_Sequence = sorted(Numberlist_pairs, key=lambda pair: pair[1])

    for pair in Number_Sequence:
        Cur = pair[0]
        Flag = 0
        Left_List = []; Right_List = []
        if not Flag:
            for i in range(0, Cur):
                Left_List.append((Cur - 1 - i, Numberlist[Cur - 1 - i]))
                if sum(item for index, item in Left_List) == Numberlist[Cur]:
                    Flag = 1
                    break
                if Numberlist[Cur - 1 - i] > Numberlist[Cur]:
                    break
        if not Flag:
            for i in range(Cur+1, len(Numberlist)):
                Right_List.append((i, Numberlist[i]))
                if sum(item for index, item in Right_List) == Numberlist[Cur]:
                    Flag = 2
                    break
                if Numberlist[i] > Numberlist[Cur]:
                    break
        if Flag == 1:
            for Del in Left_List:
                Numberlist[Del[0]] = 0
        if Flag == 2:
            for Del in Right_List:
                Numberlist[Del[0]] = 0
    Result = [item for item in Numberlist if item != 0]

    return sum(Result)