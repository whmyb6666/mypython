def remove_duplicate_sublists(lst):
    # 使用集合去除重复的子列表，然后转换回列表
    return [list(x) for x in set(tuple(sublist) for sublist in lst)]


# 示例
original_list = [[1, 2, 3], [4, 5], [1, 2, 3], [6], [4, 5], [6],[3,3]]
cleaned_list = remove_duplicate_sublists(original_list)
print(cleaned_list)  # 输出: [[1, 2, 3], [4, 5], [6]]
print(len(cleaned_list))


lst=original_list
cleaned_list=list(filter(lambda x:lst.count(x)==1,lst))
print(cleaned_list,len(cleaned_list))