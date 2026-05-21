# since each function's repr_func calls the __repr__ function of its arguments,
# format_expr will be called post-order on every internal argument's string


def format_expr(s: str) -> str:
    if len(s) <= 2:
        return s
    start = s.find("(")
    end = s.rfind(")")
    if start == -1 or end == -1:
        return s
    l = start
    r = end
    while l <= r:
        left = s[l]
        right = s[r]
        if left != "(" or right != ")":
            break
        l += 1
        r -= 1
    beginning = s[: start + 1]
    mid = s[l : r + 1]
    ending = s[end:]
    return f"{beginning}{mid}{ending}"
