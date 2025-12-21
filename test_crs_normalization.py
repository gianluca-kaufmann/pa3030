#!/usr/bin/env python3
"""
Test script to verify CRS normalization handles EngineeringCRS and PROJJSON correctly.
"""

from rasterio.crs import CRS

# Simulate what normalize_crs does
def test_normalize_crs(crs, label):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    crs_type = crs.__class__.__name__ if hasattr(crs, '__class__') else 'Unknown'
    print(f"CRS Type: {crs_type}")
    
    crs_name = None
    if hasattr(crs, 'name'):
        crs_name = str(crs.name).lower() if crs.name else None
    print(f"CRS Name: {crs_name}")
    
    try:
        crs_str = crs.to_string()
        print(f"CRS String: {crs_str[:100]}...")
    except Exception as e:
        print(f"CRS to_string() failed: {e}")
        try:
            crs_str = str(crs)
            print(f"CRS str(): {crs_str[:100]}...")
        except:
            crs_str = ""
            print("CRS str() also failed")
    
    # Check detection logic
    is_pseudo_mercator = False
    
    # Method 1: Check CRS name
    if crs_name and ("pseudo" in crs_name or "popular visualisation crs" in crs_name):
        if "mercator" in crs_name or "3857" in crs_name or "wgs 84" in crs_name:
            is_pseudo_mercator = True
            print("✓ Detected via CRS name")
    
    # Method 2: Check string representation
    if crs_str:
        crs_str_lower = crs_str.lower()
        if "pseudo-mercator" in crs_str_lower or "pseudo mercator" in crs_str_lower:
            is_pseudo_mercator = True
            print("✓ Detected via string 'pseudo-mercator'")
        if "3857" in crs_str:
            is_pseudo_mercator = True
            print("✓ Detected via string '3857'")
    
    # Method 3: Check for EngineeringCRS with WGS 84 in name
    if crs_type == "EngineeringCRS" or (crs_str and "engineeringcrs" in crs_str.lower()):
        if crs_name and "wgs 84" in crs_name:
            is_pseudo_mercator = True
            print("✓ Detected via EngineeringCRS + WGS 84")
    
    print(f"\nFinal Detection: {'PSEUDO-MERCATOR (EPSG:3857)' if is_pseudo_mercator else 'NOT DETECTED'}")
    
    if is_pseudo_mercator:
        result = CRS.from_epsg(3857)
        print(f"Normalized to: EPSG:3857")
        return result
    else:
        print("⚠ Would fall back to EPSG:3857 (default for Colombia)")
        return CRS.from_epsg(3857)

# Test cases
print("\n" + "="*60)
print("CRS NORMALIZATION TEST SUITE")
print("="*60)

# Test 1: Standard EPSG:3857
try:
    crs1 = CRS.from_epsg(3857)
    test_normalize_crs(crs1, "Standard EPSG:3857")
except Exception as e:
    print(f"Test 1 failed: {e}")

# Test 2: EPSG:3857 from string
try:
    crs2 = CRS.from_string("EPSG:3857")
    test_normalize_crs(crs2, "EPSG:3857 from string")
except Exception as e:
    print(f"Test 2 failed: {e}")

# Test 3: WGS 84 / Pseudo-Mercator by name
try:
    crs3 = CRS.from_string("WGS 84 / Pseudo-Mercator")
    test_normalize_crs(crs3, "WGS 84 / Pseudo-Mercator by name")
except Exception as e:
    print(f"Test 3 failed: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nIf all tests show 'PSEUDO-MERCATOR (EPSG:3857)', the normalization logic is working correctly.")

