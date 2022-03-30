import numpy as np

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])

    # Create 2d grid using numpy broadcasting
    XX, YY = np.atleast_2d(x, y)
    YY = YY.T  # transpose to allow broadcasting
    ZZ = XX + YY

    # Create 2d grid using meshgrid function
    XX_, YY_ = np.meshgrid(x, y)
    ZZ_ = XX_ + YY_

    # Create 2d grid without using linspace
    XX_2, YY_2 = np.ogrid[1:4:1, 10:40:10]
    ZZ_2 = XX_2 + YY_2
    ZZ_2 = ZZ_2.T

    # Create 2d grid using ogrid (does not require linspace)
    XX_2, YY_2 = np.ogrid[1:4:1, 10:40:10]
    ZZ_2 = XX_2 + YY_2
    ZZ_2 = ZZ_2.T

    # Create 2d grid using mgrid (does not require linspace)
    XX_3, YY_3 = np.ogrid[1:4:1, 10:40:10]
    ZZ_3 = XX_3 + YY_3
    ZZ_3 = ZZ_3.T

    print('x')
