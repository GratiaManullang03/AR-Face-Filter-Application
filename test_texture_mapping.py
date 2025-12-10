#!/usr/bin/env python3
"""
Test script for Face Mesh Texture Mapping.
Verifies triangulation and UV mapping implementation.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.mesh_renderer import MeshRenderer, create_texture_landmarks_from_image


def test_triangulation():
    """Test that triangulation was properly generated."""
    print("=" * 60)
    print("TEST 1: Triangulation Generation")
    print("=" * 60)

    triangles = config.FACE_MESH_TRIANGLES
    print(f"✓ Generated {len(triangles)} triangles from FACEMESH_TESSELATION")

    # Verify triangles are valid
    if len(triangles) > 0:
        print(f"✓ Sample triangles: {triangles[:3]}")

        # Check all indices are valid (0-467 for 468 landmarks)
        max_idx = max(max(tri) for tri in triangles)
        min_idx = min(min(tri) for tri in triangles)
        print(f"✓ Index range: {min_idx} to {max_idx} (expected 0-467)")

        if max_idx < 468 and min_idx >= 0:
            print("✓ All triangle indices are valid!")
        else:
            print("✗ ERROR: Invalid triangle indices!")
            return False
    else:
        print("✗ ERROR: No triangles generated!")
        return False

    return True


def test_texture_detection():
    """Test face detection in texture images."""
    print("\n" + "=" * 60)
    print("TEST 2: Texture Face Detection")
    print("=" * 60)

    assets_dir = Path(__file__).parent / "assets"

    # Test with masculine texture
    masculine_path = assets_dir / "faceMasculine.jpg"
    if masculine_path.exists():
        print(f"\nTesting: {masculine_path.name}")
        landmarks = create_texture_landmarks_from_image(str(masculine_path))

        if landmarks is not None:
            print(f"✓ Detected {len(landmarks)} landmarks")
            print(f"✓ Landmark shape: {landmarks.shape}")
            print(f"✓ Sample landmarks: {landmarks[:3]}")
        else:
            print("✗ ERROR: Could not detect face in texture!")
            return False
    else:
        print(f"✗ WARNING: {masculine_path} not found")

    # Test with feminine texture
    feminine_path = assets_dir / "faceFeminine.jpg"
    if feminine_path.exists():
        print(f"\nTesting: {feminine_path.name}")
        landmarks = create_texture_landmarks_from_image(str(feminine_path))

        if landmarks is not None:
            print(f"✓ Detected {len(landmarks)} landmarks")
        else:
            print("✗ WARNING: Could not detect face in feminine texture")

    return True


def test_mesh_renderer():
    """Test the MeshRenderer class."""
    print("\n" + "=" * 60)
    print("TEST 3: MeshRenderer Initialization")
    print("=" * 60)

    # Create a simple test texture
    test_texture = np.ones((500, 500, 3), dtype=np.uint8) * 128

    renderer = MeshRenderer(test_texture, debug_mode=False)
    print(f"✓ MeshRenderer initialized")
    print(f"✓ Texture size: {renderer.texture_w}x{renderer.texture_h}")
    print(f"✓ Triangle count: {len(renderer.triangles)}")

    return True


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration Loading")
    print("=" * 60)

    print(f"✓ Texture masks defined: {len(config.TEXTURE_MASKS)}")

    for name, mask_config in config.TEXTURE_MASKS.items():
        print(f"\n  {name}:")
        print(f"    - Path: {mask_config.asset_path.name}")
        print(f"    - Opacity: {mask_config.opacity}")
        print(f"    - Wireframe: {mask_config.debug_wireframe}")
        print(f"    - Subsample: {mask_config.subsample}")
        print(f"    - Exists: {mask_config.asset_path.exists()}")

    return True


def visualize_triangulation():
    """Create a visualization of the triangulation."""
    print("\n" + "=" * 60)
    print("TEST 5: Triangulation Visualization")
    print("=" * 60)

    try:
        import mediapipe as mp

        # Create a test image
        test_img = np.ones((600, 600, 3), dtype=np.uint8) * 255

        # Create normalized landmarks in a grid for visualization
        landmarks = []
        size = 600
        margin = 100
        for i in range(468):
            # Distribute points evenly
            x = margin + (i % 20) * (size - 2 * margin) // 20
            y = margin + (i // 20) * (size - 2 * margin) // 24
            landmarks.append([x, y])

        landmarks = np.array(landmarks, dtype=np.float32)

        # Draw triangles
        triangles = config.FACE_MESH_TRIANGLES[:50]  # Draw first 50 for visibility
        for tri in triangles:
            pt1 = tuple(landmarks[tri[0]].astype(int))
            pt2 = tuple(landmarks[tri[1]].astype(int))
            pt3 = tuple(landmarks[tri[2]].astype(int))

            cv2.line(test_img, pt1, pt2, (0, 255, 0), 1)
            cv2.line(test_img, pt2, pt3, (0, 255, 0), 1)
            cv2.line(test_img, pt3, pt1, (0, 255, 0), 1)

        output_path = Path(__file__).parent / "triangulation_test.jpg"
        cv2.imwrite(str(output_path), test_img)
        print(f"✓ Saved triangulation visualization to: {output_path}")

    except Exception as e:
        print(f"✗ Could not create visualization: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FACE MESH TEXTURE MAPPING - TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        ("Triangulation Generation", test_triangulation),
        ("Texture Detection", test_texture_detection),
        ("MeshRenderer", test_mesh_renderer),
        ("Configuration", test_config_loading),
        ("Visualization", visualize_triangulation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The implementation is ready.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
