#canty module for Google Earth Engine
import ee
"""
This module contains functions sourced from Dr. Mort Canty's tutorial on change detection using the MAD transformation. For implementation. The module needs to be in the same located in the same code as the code for executing the module. 
The Tutorial is available at: https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt2
"""
ee.Authenticate()
# Initialize the library.
ee.Initialize(project='project') # for intializing the project and acessing the API,
import geemap
import numpy as np
import random, time
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from pprint import pprint  # for pretty printing
#########################################################
# MAD transformation
# Enter your own export to assets path name here -----------
EXPORT_PATH = 'projects/ee-thesiswar/assets/imad/'
print(EXPORT_PATH)
# ------------------------------------------------
def trunc(values, dec = 3):
    '''Truncate a 1-D array to dec decimal places.'''
    return np.trunc(values*10**dec)/(10**dec)
# Display an image in a one percent linear stretch.
def covarw(image, weights=None, scale=20, maxPixels=1e10):
    '''
    Return the centered image and its weighted covariance matrix.

    Parameters:
    - image: The input image.
    - weights: The weights to be used for the covariance calculation. If not provided, a constant weight of 1 will be used.
    - scale: The scale at which to compute the covariance. Default is 20 scaling here allowed for comparison of images with different resoltuons
    - maxPixels: The maximum number of pixels to compute the covariance. Default is 1e10.

    Returns:
    A tuple containing the centered image and its weighted covariance matrix.
    '''

    try:
        # Get geometry, band names, and number of bands.
        geometry = image.geometry()
        bandNames = image.bandNames()
        N = bandNames.length()

        # If weights are not provided, use a constant weight of 1.
        if weights is None:
            weights = image.constant(1)

        # Create an image with band names and weights.
        weightsImage = image.multiply(ee.Image.constant(0)).add(weights)

        # Compute means and centered image.
        means = image.addBands(weightsImage) \
                    .reduceRegion(ee.Reducer.mean().repeat(N).splitWeights(),
                                scale=scale,
                                maxPixels=maxPixels) \
                    .toArray() \
                    .project([1])
        centered = image.toArray().subtract(means)

        # Compute weighted covariance matrix.
        B1 = centered.bandNames().get(0)
        b1 = weights.bandNames().get(0)
        nPixels = ee.Number(centered.reduceRegion(ee.Reducer.count(),
                                                scale=scale,
                                                maxPixels=maxPixels).get(B1))
        sumWeights = ee.Number(weights.reduceRegion(ee.Reducer.sum(),
                                                    geometry=geometry,
                                                    scale=scale,
                                                    maxPixels=maxPixels).get(b1))
        covw = centered.multiply(weights.sqrt()) \
                    .toArray() \
                    .reduceRegion(ee.Reducer.centeredCovariance(),
                                    geometry=geometry,
                                    scale=scale,
                                    maxPixels=maxPixels) \
                    .get('array')
        covw = ee.Array(covw).multiply(nPixels).divide(sumWeights)

        return (centered.arrayFlatten([bandNames]), covw)

    except Exception as e:
        print('Error: %s' % e)
def corr(cov):
    '''
    Transfrom covariance matrix to correlation matrix.

    Parameters:
    - cov: The covariance matrix.

    Returns:
    The correlation matrix.
    '''

    # Diagonal matrix of inverse sigmas.
    sInv = cov.matrixDiagonal().sqrt().matrixToDiag().matrixInverse()

    # Transform.
    corr = sInv.matrixMultiply(cov).matrixMultiply(sInv).getInfo()

    # Truncate.
    return [list(map(trunc, corr[i])) for i in range(len(corr))]

def geneiv(C, B):
    '''
    Return the eigenvalues and eigenvectors of the generalized eigenproblem

    Parameters:
    - C: A 2D array representing the matrix C.
    - B: A 2D array representing the matrix B.

    Returns:
    - A tuple (lambdas, eigenvecs) containing the eigenvalues and eigenvectors.
    Raises:
    - Any exception that occurs during the computation.

    Example usage:
    C = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    lambdas, eigenvecs = geneiv(C, B)
    print("Eigenvalues:", lambdas)
    print("Eigenvectors:", eigenvecs)
    '''

    try:
        # Convert input arrays to Earth Engine Arrays.
        C = ee.Array(C)
        B = ee.Array(B)

        # Compute the inverse of the Cholesky decomposition of B (Li = choldc(B)^-1).
        Li = ee.Array(B.matrixCholeskyDecomposition().get('L')).matrixInverse()

        # Solve the symmetric, ordinary eigenproblem Li*C*Li^T*x = lambda*x.
        Xa = Li.matrixMultiply(C) \
            .matrixMultiply(Li.matrixTranspose()) \
            .eigen()

        # Extract the eigenvalues as a row vector.
        lambdas = Xa.slice(1, 0, 1).matrixTranspose()

        # Extract the eigenvectors as columns.
        X = Xa.slice(1, 1).matrixTranspose()

        # Compute the generalized eigenvectors as columns by multiplying Li^T with X.
        eigenvecs = Li.matrixTranspose().matrixMultiply(X)

        # Return the eigenvalues and eigenvectors as a tuple.
        return (lambdas, eigenvecs)

    except Exception as e:
        print('Error: %s' % e)
def mad_run(image1, image2, scale=20):
    '''
    The MAD transformation of two multiband images.

    Parameters:
    - image1: The first multiband image.
    - image2: The second multiband image.
    - scale: The scale at which to compute the covariance. Default is 20.

    Returns:
    A tuple containing the U, V, MAD, and Z images
Raises:
    - Any exception that occurs during the computation.

    '''
    try:
        # Combine two images into one
        image = image1.addBands(image2)
        # Get the number of bands.
        nBands = image.bandNames().length().divide(2)
        # Compute the centered image and its weighted covariance matrix.
        centeredImage, covarArray = covarw(image, scale=scale)
        # Extract band names for the two sets of bands.
        bNames = centeredImage.bandNames()
        bNames1 = bNames.slice(0, nBands)
        bNames2 = bNames.slice(nBands)
        # Select bands for the two centered images.
        centeredImage1 = centeredImage.select(bNames1)
        centeredImage2 = centeredImage.select(bNames2)
        s11 = covarArray.slice(0, 0, nBands).slice(1, 0, nBands)
        s22 = covarArray.slice(0, nBands).slice(1, nBands)
        s12 = covarArray.slice(0, 0, nBands).slice(1, nBands)
        s21 = covarArray.slice(0, nBands).slice(1, 0, nBands)
        # Calculate matrices for generalized eigenproblems.
        c1 = s12.matrixMultiply(s22.matrixInverse()).matrixMultiply(s21)
        b1 = s11
        c2 = s21.matrixMultiply(s11.matrixInverse()).matrixMultiply(s12)
        b2 = s22
        # Solution of generalized eigenproblems.
        lambdas, A = geneiv(c1, b1)
        _, B = geneiv(c2, b2)
        rhos = lambdas.sqrt().project(ee.List([1]))
        # MAD variances.
        sigma2s = rhos.subtract(1).multiply(-2).toList()
        sigma2s = ee.Image.constant(sigma2s)
        # Ensure sum of positive correlations between X and U is positive.
        tmp = s11.matrixDiagonal().sqrt()
        ones = tmp.multiply(0).add(1)
        tmp = ones.divide(tmp).matrixToDiag()
        s = tmp.matrixMultiply(s11).matrixMultiply(A).reduce(ee.Reducer.sum(), [0]).transpose()
        A = A.matrixMultiply(s.divide(s.abs()).matrixToDiag())
        # Ensure positive correlation.
        tmp = A.transpose().matrixMultiply(s12).matrixMultiply(B).matrixDiagonal()
        tmp = tmp.divide(tmp.abs()).matrixToDiag()
        B = B.matrixMultiply(tmp)
        # Canonical and MAD variates as images.
        centeredImage1Array = centeredImage1.toArray().toArray(1)
        centeredImage2Array = centeredImage2.toArray().toArray(1)
        U = ee.Image(A.transpose()).matrixMultiply(centeredImage1Array) \
                    .arrayProject([0]) \
                    .arrayFlatten([bNames2])
        V = ee.Image(B.transpose()).matrixMultiply(centeredImage2Array) \
                    .arrayProject([0]) \
                    .arrayFlatten([bNames2])
        MAD = U.subtract(V)
        # Chi-square image.
        Z = MAD.pow(2) \
               .divide(sigma2s) \
               .reduce(ee.Reducer.sum())
        return (U, V, MAD, Z)
    except Exception as e:
        print('Error: %s' % e)

 #########################################
# iMAD functions
def chi2cdf(Z, df):
        """
        Calculate the cumulative distribution function (CDF) of the chi-square distribution.

        Parameters:
        Z (ee.Image): The input image representing the chi-square random variable. The sum of squared Standardized MAD variates.
        df (int): The degrees of freedom of the chi-square distribution.Number of Bands - 1
        Returns:
        ee.Image: The image representing the CDF of the chi-square distribution.

        Notes:
        - The chi-square distribution is a continuous probability distribution that arises in the context of
            hypothesis testing and confidence interval estimation for the variance of a normally distributed population.
        - The CDF of the chi-square distribution gives the probability that a chi-square random variable is less than or equal to a given value.

        """
        return ee.Image(Z.divide(2)).gammainc(ee.Number(df).divide(2))
def imad(current,prev):
    '''
    Iterator function for iMAD.

    Parameters:
    - current: The current iteration value.
    - prev: The previous iteration valu
    - returns done
    '''
    done =  ee.Number(ee.Dictionary(prev).get('done'))
    return ee.Algorithms.If(done, prev, imad1(current, prev))

def imad1(current,prev):
    '''
    Iteratively re-weighted MAD.

    Parameters:
    - current: The current iteration value.
    - prev: The previous iteration value.

    Returns:
    The updated iteration value.
    '''
    image = ee.Image(ee.Dictionary(prev).get('image'))
    Z = ee.Image(ee.Dictionary(prev).get('Z'))
    allrhos = ee.List(ee.Dictionary(prev).get('allrhos'))
    nBands = image.bandNames().length().divide(2)
    weights = chi2cdf(Z,nBands).subtract(1).multiply(-1)
    scale = ee.Dictionary(prev).getNumber('scale')
    niter = ee.Dictionary(prev).getNumber('niter')
    # Weighted stacked image and weighted covariance matrix.
    centeredImage, covarArray = covarw(image, weights, scale)
    bNames = centeredImage.bandNames()
    bNames1 = bNames.slice(0, nBands)
    bNames2 = bNames.slice(nBands)
    centeredImage1 = centeredImage.select(bNames1)
    centeredImage2 = centeredImage.select(bNames2)
    s11 = covarArray.slice(0, 0, nBands).slice(1, 0, nBands)
    s22 = covarArray.slice(0, nBands).slice(1, nBands)
    s12 = covarArray.slice(0, 0, nBands).slice(1, nBands)
    s21 = covarArray.slice(0, nBands).slice(1, 0, nBands)
    c1 = s12.matrixMultiply(s22.matrixInverse()).matrixMultiply(s21)
    b1 = s11
    c2 = s21.matrixMultiply(s11.matrixInverse()).matrixMultiply(s12)
    b2 = s22
    # Solution of generalized eigenproblems.
    lambdas, A = geneiv(c1, b1)
    _, B       = geneiv(c2, b2)
    rhos = lambdas.sqrt().project(ee.List([1]))
    # Test for convergence.
    lastrhos = ee.Array(allrhos.get(-1))
    done = rhos.subtract(lastrhos) \
               .abs() \
               .reduce(ee.Reducer.max(), ee.List([0])) \
               .lt(ee.Number(0.0001)) \
               .toList() \
               .get(0)
    allrhos = allrhos.cat([rhos.toList()])
    # MAD variances.
    sigma2s = rhos.subtract(1).multiply(-2).toList()
    sigma2s = ee.Image.constant(sigma2s)
    # Ensure sum of positive correlations between X and U is positive.
    tmp = s11.matrixDiagonal().sqrt()
    ones = tmp.multiply(0).add(1)
    tmp = ones.divide(tmp).matrixToDiag()
    s = tmp.matrixMultiply(s11).matrixMultiply(A).reduce(ee.Reducer.sum(), [0]).transpose()
    A = A.matrixMultiply(s.divide(s.abs()).matrixToDiag())
    # Ensure positive correlation.
    tmp = A.transpose().matrixMultiply(s12).matrixMultiply(B).matrixDiagonal()
    tmp = tmp.divide(tmp.abs()).matrixToDiag()
    B = B.matrixMultiply(tmp)
    # Canonical and MAD variates.
    centeredImage1Array = centeredImage1.toArray().toArray(1)
    centeredImage2Array = centeredImage2.toArray().toArray(1)
    U = ee.Image(A.transpose()).matrixMultiply(centeredImage1Array) \
                   .arrayProject([0]) \
                   .arrayFlatten([bNames1])
    V = ee.Image(B.transpose()).matrixMultiply(centeredImage2Array) \
                   .arrayProject([0]) \
                   .arrayFlatten([bNames2])
    iMAD = U.subtract(V)
    # Chi-square image.
    Z = iMAD.pow(2) \
              .divide(sigma2s) \
              .reduce(ee.Reducer.sum())
    return ee.Dictionary({'done': done, 'scale': scale, 'niter': niter.add(1),
                          'image': image, 'allrhos': allrhos, 'Z': Z, 'iMAD': iMAD})

##########################################################
# to run imad
def run_imad(aoi, image1, image2, assetFN, scale=20, maxiter=100,):
    """
    Run the iMAD algorithm on two input images.

    Parameters:
    - aoi: Area of interest (ee.Geometry) to clip the output image.
    - image1: First input image (ee.Image).
    - image2: Second input image (ee.Image).
    - assetFN: Filename for exporting the iMAD result as an asset (str).
    - scale: Scale for the analysis (int, default=20).
    - maxiter: Maximum number of iterations for the iMAD algorithm (int, default=100).Elsewise, the algorithm will stop when the change in the MAD variances is less than 0.0001.
    """
    try:
        # Get the number of bands in the first input image
        N = image1.bandNames().length().getInfo()
        # Create a list of names for the iMAD images and the Z image
        imadnames = ['iMAD'+str(i+1) for i in range(N)]
        imadnames.append('Z')
        # Create a list of numbers from 1 to maxiter for iteration
        inputlist = ee.List.sequence(1, maxiter)
        # Create the initial dictionary for the first iteration
        first = ee.Dictionary({'done':0,
                            'scale': scale,
                            'niter': ee.Number(0),
                            'image': image1.addBands(image2),
                            'allrhos': [ee.List.sequence(1, N)],
                            'Z': ee.Image.constant(0),
                            'iMAD': ee.Image.constant(0)})
        # Iterate through the list of numbers using the imad function
        result = ee.Dictionary(inputlist.iterate(imad, first))
        # Retrieve the results from the iteration
        iMAD = ee.Image(result.get('iMAD')).clip(aoi)
        rhos = ee.String.encodeJSON(ee.List(result.get('allrhos')).get(-1))
        Z = ee.Image(result.get('Z'))
        niter = result.getNumber('niter')
        # Create an iMAD export image with the iMAD and Z bands
        iMAD_export = ee.Image.cat(iMAD, Z).rename(imadnames).set('rhos', rhos, 'niter', niter)
        # Export the iMAD image to an asset
        assetId = EXPORT_PATH + assetFN
        assexport = ee.batch.Export.image.toAsset(iMAD_export,
                        description='assetExportTask',
                        assetId=assetId, scale=scale, maxPixels=1e10)
        assexport.start()
        # Print the export information
        print('Exporting iMAD to %s\n task id: %s'%(assetId, str(assexport.id)))
    except Exception as e:
        print('Error: %s'%e)
