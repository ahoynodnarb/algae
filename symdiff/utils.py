def remove_redudant_parens(s: str) -> str:
    if len(s) <= 2:
        return ""
    start = s.find("(")
    end = s.rfind(")")
    l = start
    r = end
    while l <= r:
        left = s[l]
        right = s[r]
        if left != "(" or right != ")":
            break
        l += 1
        r -= 1
    beginning = s[:start]
    mid = s[l : r + 1]
    ending = s[end + 1 :]
    return f"{beginning}({mid}){ending}"
