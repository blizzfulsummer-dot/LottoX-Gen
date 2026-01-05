def analyze_numbers(nums):
    if len(nums) != 6:
        raise ValueError("Input must contain exactly 6 numbers")

    results = {}

    # Odd and Even counts
    results["Odd"] = sum(1 for n in nums if n % 2 != 0)
    results["Even"] = sum(1 for n in nums if n % 2 == 0)

    # Consecutive differences
    diffs = [abs(nums[i+1] - nums[i]) for i in range(5)]

    # Exact gap checks
    results["2Jump"] = diffs.count(2)
    results["3Jump"] = diffs.count(3)
    results["5Jump"] = diffs.count(5)
    results["10Jump"] = diffs.count(10)

    # Series (consecutive numbers)
    results["2Series"] = sum(1 for i in range(5) if nums[i+1] - nums[i] == 1)
    results["3Series"] = sum(1 for i in range(4) if nums[i+1] - nums[i] == 1 and nums[i+2] - nums[i+1] == 1)

    # Pattern list (fixed order)
    pattern = [
        results["Odd"],
        results["Even"],
        results["2Jump"],
        results["3Jump"],
        results["5Jump"],
        results["10Jump"],
        results["2Series"],
        results["3Series"]
    ]

    return results, pattern


# Example
numbers = [5, 6, 8, 11, 16, 26]
res, pat = analyze_numbers(numbers)
print(res)
print(pat)

