
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


def get_merged_row_list(sorted_row_list):
    merged_row_list = []
    for row in sorted_row_list:
        left = min(row, key = lambda x : x[0][0])[0][0]
        top = min(row, key = lambda x : x[0][1])[0][1]
        right = max(row, key = lambda x : x[1][0])[1][0]
        bottom = max(row, key = lambda x : x[1][1])[1][1]
        merged_row_list.append([(left, top),(right, bottom)])
    
    return merged_row_list
