from athena_cost_guard import pricing


def test_ten_mb_minimum_applied():
    # A tiny scan is billed as the 10 MB minimum.
    assert pricing.billable_bytes(1) == pricing.MIN_BYTES


def test_rounds_up_to_nearest_ten_mb():
    assert pricing.billable_bytes(11 * 1000 * 1000) == 20 * 1000 * 1000


def test_one_tb_costs_five_dollars():
    cost = pricing.cost_usd(pricing.BYTES_PER_TB, price_per_tb=5.0)
    assert round(cost, 6) == 5.0


def test_region_lookup_falls_back_to_default():
    assert pricing.price_for_region("mars-central-1") == pricing.DEFAULT_PRICE_PER_TB
    assert pricing.price_for_region("us-east-1") == 5.0
    assert pricing.price_for_region(None) == pricing.DEFAULT_PRICE_PER_TB
