
def get_row_list(lt_rb_list, threshold=3):
    row_groups = []
    for box in lt_rb_list:
        l, t, r, b = box[0][0], box[0][1], box[1][0], box[1][1]
        if row_groups == []:
            row_groups.append([box])
        else:
            if abs(row_groups[-1][-1][0][1]-t) < threshold and abs(row_groups[-1][-1][1][1] - b)< threshold:
                row_groups[-1].append(box)
            else:
                row_groups.append([box])
    sorted_row_groups = [sorted(row, key = lambda x : x[0]) for row in row_groups]

    return sorted_row_groups

def fix_format(sorted_row_list, margin=0):
    result_list = []
    for row in sorted_row_list:
        for box in row: ##   l t r b
            l = box[0][0]-margin
            t = box[0][1]-margin
            r = box[1][0]+margin
            b = box[1][1]+margin

            if l<0: l = 0
            if t<0 : t = 0
            result_list.append([(l, t), (r, b)])
    
    return result_list
        


    return 
def get_merged_row_list(sorted_row_list, margin=0):
    merged_row_list = []
    for row in sorted_row_list:
        left = min(row, key = lambda x : x[0][0])[0][0] - margin
        top = min(row, key = lambda x : x[0][1])[0][1] - margin
        right = max(row, key = lambda x : x[1][0])[1][0] + margin
        bottom = max(row, key = lambda x : x[1][1])[1][1] + margin

        if left<0: left = 0
        if top<0: top = 0


        merged_row_list.append([(left, top),(right, bottom)])
    
    return merged_row_list
