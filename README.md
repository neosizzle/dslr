# /dev/log for DSLR
The goal of this project is to build a classification algorithm using the one vs rest machine learning algorithm; as well as to build some intuition on the basics of data science by introducing a problem from the Harry Potter series - the sorting hat.

We are given a training dataset of students scores in multiple subjects as well as their assigned houses. We will be using this dataset to create a logistic regression model to predict the houses of some other students, as they provide their own score.

It is assumed to have basic understanding of linear regression at this point.

# Maximum likelihood for normal distribution

$$
P\left( x\right) =\dfrac{1}{\sqrt{2\pi \sigma ^{2}}}e^{-\left( x-\mu\right) ^{2}/2\sigma ^{2}}
$$


$\sigma$ refers to standard deviation, and $\mu$ refers to mean. 

Above is the formula for a normal distribution / Gaussian curve for some data $x$. The standard deviation will affect the width of the curve and the mean affects the position.

> If we adapt a probalistic view in linear regression, the maximum likelihood estiator assuming zero mean gaussian noise is same as linear regression with squared error.

The goal of maximum likelihood estimation is to fit the parameters (mean and stddev) of the Gaussian curve so that all observed data has the biggest probability of happening.

![image](https://hackmd.io/_uploads/r1ka4okQC.png)

We go back to the formula that gives us the probability of a single x:

$$
P\left( x\right) =\dfrac{1}{\sqrt{2\pi \sigma ^{2}}}e^{-\left( x-\mu\right) ^{2}/2\sigma ^{2}}
$$

to calculate all the probabilities for $x$, we can use the product on the return values of the $P\left( x;\mu ,\sigma \right)$ for all $x$.

[insert maximize gaussian here...]

$$
P\left( x;\mu,\sigma \right) = P\left( x_{0};\mu,\sigma \right) \times P\left( x_{1};\mu,\sigma \right)  \times \ldots P\left( x_{n};\mu,\sigma \right) 
$$

And if we want to maximize a function like that, we have to use differientiation. The original roduct form is hard to differienciate. hence we would first transform the function to its natural logarithmic counterpart $\log _{e}$. This is fine because $\log _{e}$ is a monotomically incresing function. (x increases, y also increases albeit at a different rate)

![image](https://hackmd.io/_uploads/SkjpIikQ0.png)


and it transforms to:

$$
\ln \left[ P\left( x;\mu,\sigma\right) \right] = \ln\left[ \dfrac{-1}{\sqrt[\sigma ] {2\pi }}-\dfrac{\left( x_{0}-\mu \right) ^{2}}{2\sigma ^{2}}\right] + \ln\left[ \dfrac{-1}{\sqrt[\sigma ] {2\pi }}-\dfrac{\left( x_{1}-\mu \right) ^{2}}{2\sigma ^{2}}\right] \times \ldots
$$

say we have the $x$, as 9, 9.5, 11; the transformed equation gives us 

$$
\ln \left[ P\left( x;\mu,\sigma\right) \right] = \ln\left[ \dfrac{-1}{\sqrt[\sigma ] {2\pi }}-\dfrac{\left( 9-\mu \right) ^{2}}{2\sigma ^{2}}\right] + \ln\left[ \dfrac{-1}{\sqrt[\sigma ] {2\pi }}-\dfrac{\left( 9.5-\mu \right) ^{2}}{2\sigma ^{2}}\right] + \ln\left[ \dfrac{-1}{\sqrt[\sigma ] {2\pi }}-\dfrac{\left( 11-\mu \right) ^{2}}{2\sigma ^{2}}\right]
$$

we can simplify this equation using logarithm law to 

$$
3\ln \left( \sigma \right) -\dfrac{3}{2}\ln \left( \sigma \right) -\dfrac{1}{2\sigma ^{2}}\left[ \left( 9-\mu\right) ^{2}+\left( 9.5-\mu\right) ^{2}+\left( 11-\mu\right) ^{-2}\right]
$$

We cancalculate the partial derivative of the above function in terms of $\mu$ to find the value of $\mu$ where the function above is maximum

$$
\frac{\partial}{\partial u}=\dfrac{1}{2\sigma ^{2}}\left[ \dfrac{\partial }{\partial \mu}\left( 3u\right) ^{2}-\dfrac{\partial }{\partial\mu}\left( 59\mu\right) +\dfrac{\partial }{\partial u}292.25\right] 
$$

$$
=> \dfrac{1}{2\sigma ^{2}}\left( 6\mu-59\right) 
$$

$$
\dfrac{1}{2\sigma ^{2}}\left( 6\mu-59\right) = 0
$$

$$
\mu = 9.83
$$

> The standard deviation can also be calculated in a similar manner

# Logistic regression
Unlike linear regression, logistic regression predicts wether something is **True** or **false** (1 or 0) and the line we fit is an 'S' shape like so

![image](https://hackmd.io/_uploads/B1nhx4e7A.png)
If we want to predict new data, the line will be mapped to the probability of the data being classified as true. This can be used for calssification

Since logistic regression does not have least squares or R-sqaured, we will use the maximum likelihood of all ovserved data to fit the regression line

# Logistic regression coefficients
Coefficients in linear regression are the $m$ and $c$ values. They are constants that defined the regression line.

For logistic regression, we also have $m$ and $c$ as our coefficients. Although the regression line itself isnt a straight line, we are able to apply the log of odds ($log(1/1-p)$) function to all values of $y$ to get a straight line.

![image](https://hackmd.io/_uploads/S1E-Q4g7R.png)

> These coefficients only apply for continious variables (values of $x$ is a range), not for discrete variables (only contain a set number of values)

# Maximum likelihood in logistic regression
Toget coefficients for logistic regression represented in a linear nammer, we cant use least squares because the R square will be infinity. Instead we can use the maximum likelihood by doing the following. 

1. Draw a best fitting candidate straight line
2. Project the original data points to the candidate line to obtain the  log(odds) of said data. 
3. Transform the log(odds) back into normal probabilities by using the formula $p=e^{\log \left( odds\right) }/ 1 + e^{\log \left( odds\right) }$ (Reorder straight projection into curved projection)
4. We then calculate the total likelihood of the values for the true state using the product of their probabilities
5. We then calculate the total likelihood of the false state using the product of (1- their probabilities)

![image](https://hackmd.io/_uploads/r1SLFNxmA.png)

So far we have only be able to classify values of 0 and 1 (binary). To calssify more categories like A, B, C, D, we can group them like A and not A, B and not B .... We then take the true values for each pair and assign the value to the corresponding class with the highest probability. This is called **One vs Rest classification**

# Gradient descent in logistic regression
We will start with the **sigmoid function** (function that turns inf to 1 and -inf to 0) aka the function that calculates probability

$$
y=g\left( z\right) =\dfrac{1}{1+e^{-z}}
$$

This sigmoid function also works as our hypothesis function; function that is run to make predictions


$$
h_{\theta }\left( x\right) =\dfrac{1}{1+e^{-\theta ^{T}x}}
$$

> $\theta$ is coefficients [$\theta_{0}$, $\theta_{1}$, ... $\theta_{n}$] where n are features hence,

$$
P( y=  1| \theta ,x) =\dfrac{1}{1+e^{-\theta ^{T}x}}
$$

and

$$
P( y=  0| \theta ,x) = 1 - P( y=  1| \theta ,x)
$$

> $y$ is the probability and $\theta$ is the coefficient and $x$ are the variables

The functions above can be simplified to

$$
P( y| \theta ,x) = \left[\dfrac{1}{1+e^{-\theta ^{T}x}}\right]^y \times \left[\dfrac{1}{1+e^{\theta ^{T}x}}\right]^{1-y}
$$

> Powers $1-y$ and $y$ will cancl out each other (become 0) based on the actual value of $y$

With this, we can also define the cost function for the 2 possible $y$ valeus: $-\log _{e}\left( h_{\theta }\left( x\right) \right)$ if $y$ = 1 and $\log _{e}\left( 1 - h_{\theta }\left( x\right) \right)$ if $y$ = 0.


![image](https://hackmd.io/_uploads/S1uSySlmR.png)

We can simplify the cost functions to 1 function

$$
cost(h_{\theta }\left( x\right) ,y)=-y * \log \left( h_{\theta }\left( x\right) \right) -\left( 1-y\right) * \log \left[ 1-h_{\theta }\left( x\right) \right] 
$$

For $m$ observation, we can calculate the cost as

$$
cost = -\dfrac{1}{m}\sum ^{m}_{i=1}\left[ cost(h_{\theta }\left( x_i\right) ,y_i)\right] 
$$

Since our coefficients will be $m$ and $c$ for the straight line function, we will need to get its partial derivatives from the cost function $\dfrac{\partial }{\partial m}$ and $\dfrac{\partial }{\partial c}$.

We first declare some variables:
$$
\sigma \left( x\right) =\dfrac{1}{1+e^{-x}}
$$

$$
a = \sigma \left( z\right)
$$

$$
z = mx + c
$$

We can redefine our partial derivatives using the chain rule:

$$
\dfrac{\partial }{\partial m} = \dfrac{\partial }{\partial a} \times \dfrac{\partial a}{\partial z} \times \dfrac{\partial z}{\partial m} 
$$

We start by solving for $\dfrac{\partial }{\partial a}$

$$
\dfrac{\partial }{\partial a} = \dfrac{\partial }{\partial a}(ylog(a) + (1-y)log(1-a))
$$

$$
\dfrac{\partial }{\partial a} = -\dfrac{y}{a} - (-1)\dfrac{\left( 1-y\right) }{\left( 1-a\right) }
$$


$$
\dfrac{\partial }{\partial a} = -\dfrac{y}{a}+\dfrac{\left( 1-y\right) }{\left( 1-a\right) }
$$

Solve for $\dfrac{\partial a}{\partial z}$

From previous definitions, we know that $a^{2}=\dfrac{1}{\left( 1+e^{-z}\right) ^{2}}$
which we can deduce to $e^{-z}=\dfrac{\left( 1-a\right) }{a}$

$$
\dfrac{\partial a}{\partial z} = \dfrac{\partial a}{\partial z}\left( {1}+e^{-z}\right) ^{-1}
$$

$$
\dfrac{\partial a}{\partial z} = \dfrac{-1}{\left( 1+e^{-z}\right) ^{2}}\left( e^{-z}\right) \left( -1\right) 
$$

$$
\dfrac{\partial a}{\partial z} = \dfrac{e^{-z}}{\left( 1+e^{-z}\right) ^{2}}
$$

$$
\dfrac{\partial a}{\partial z} = \dfrac{(1 - a)}{a} \times a^2
$$

$$
\dfrac{\partial a}{\partial z} = a(1 - a)
$$

Solve for $\dfrac{\partial z}{\partial m}$
$$
\dfrac{\partial a}{\partial z} = \dfrac{\partial a}{\partial z}(mx + c)
$$

$$
\dfrac{\partial a}{\partial z} = x
$$

Going back to $\dfrac{\partial }{\partial m} = \dfrac{\partial }{\partial a}\times \dfrac{\partial a}{\partial z} \times \dfrac{\partial z}{\partial m}$

$$
\dfrac{\partial }{\partial m} = \left[-\dfrac{y}{a} - (-1)\dfrac{\left( 1-y\right) }{\left( 1-a\right) } \right]\times a(1-a) \times x
$$

$$
\dfrac{\partial }{\partial m} = (a-y)x
$$

Solve for $\dfrac{\partial }{\partial c}$

$$
$\dfrac{\partial }{\partial c} = \dfrac{\partial }{\partial a} \times \dfrac{\partial a}{\partial z} \times \dfrac{\partial z}{\partial c}
$$

$$
\dfrac{\partial }{\partial c} = a - y
$$

We can use $\dfrac{\partial }{\partial c}$ and $\dfrac{\partial }{\partial m}$ in our gradient descent algorithm now.

# Data description
One of the requirements of this project is to provide a script to describe the data (give a high level view of the data itself) which can be done by evaluating different metrics that give an idea what the data is like. The description of each metric is as follows

1. Count: This metric represents how many entries in our dataset, useful for calculating other metrics like mean
2. Mean: The average value for all the values in a certain measurement (Scores of flying, astronomy ...)
3. Standard deviation: By itself, it is a unit of measurement that dictates the distance between a value to its mean in a bell curve. The magnitude of this value acts as a scale factor when we are detemrmining the exact distance instead of using standard deviation notation. Having a higher standard deviation means having more volatile changes in the data.
4. Min and Max: The minimum and maximum values for a measurement
5. Percentiles (25%, 50%, 75%): The exact value that will appear in the top (25/50/75)% when the dataset is sorted.
6. Skewness: This determines if the data is normally distributed or grouped to the upper / lower values. Data which are more grouped on the lower values have a positive skew since their tail is on the right side of the curve and vice versa.

Since the training dataset looks something like this, determing those metrics just require some basic parsing

```
Index,Hogwarts House,First Name,Last Name,Birthday,Best Hand,Arithmancy,Astronomy,Herbology,Defense Against the Dark Arts,Divination,Muggle Studies,Ancient Runes,History of Magic,Transfiguration,Potions,Care of Magical Creatures,Charms,Flying
0,Ravenclaw,Tamara,Hsu,2000-03-30,Left,58384.0,-487.88608595139016,5.727180298550763,4.8788608595139005,4.7219999999999995,272.0358314131986,532.4842261151226,5.231058287281048,1039.7882807428462,3.7903690663529614,0.7159391270136213,-232.79405,-26.89
1,Slytherin,Erich,Paredes,1999-10-14,Right,67239.0,-552.0605073421984,-5.987445780050746,5.520605073421985,-5.612,-487.3405572673422,367.7603030171392,4.107170286816076,1058.9445920642218,7.248741976146588,0.091674183916857,-252.18425,-113.45
2,Ravenclaw,Stephany,Braun,1999-11-03,Left,23702.0,-366.0761168823237,7.7250166064392305,3.6607611688232367,6.14,664.8935212343011,602.5852838484592,3.5555789956034967,1088.0883479121803,8.728530920939827,-0.5153268462809037,-227.34265,30.42
3,Gryffindor,Vesta,Mcmichael,2000-08-19,Left,32667.0,697.742808842469,-6.4972144445985505,-6.9774280884246895,4.026,-537.0011283872882,523.9821331934736,-4.8096366069645935,920.3914493107919,0.8219105005879808,-0.014040417239052931,-256.84675,200.64
4,Gryffindor,Gaston,Gibbs,1998-09-27,Left,60158.0,436.7752035539525,-7.820623052454388,,2.2359999999999998,-444.2625366004496,599.3245143172293,-3.4443765754165385,937.4347240534976,4.311065821291761,-0.2640700765443832,-256.3873,157.98
5,Slytherin,Corrine,Hammond,1999-04-04,Right,21209.0,-613.6871603822729,-4.289196726941419,6.136871603822727,-6.5920000000000005,-440.99770426820817,396.20180391410247,5.3802859494804585,1052.8451637299704,11.751212035101073,1.049894068203692,-247.94548999999998,-34.69
6,Gryffindor,Tom,Guido,2000-09-30,Left,49167.0,628.0460512248516,-4.861976240490781,-6.280460512248515,,-926.8925116349667,583.7424423327342,-7.322486416427907,923.5395732944658,1.6466661386700716,0.1530218296077356,-257.83447,261.55
...
...
...
```
 
# Data visualization
The data set provided has multiple features, and we want to pick the correct features to do logistic regression on. To have more insight on the data, we can start by visualizing the data by representing it in different formats.

We were given questions to ponder and ways to visualize the data when answering those questions. One of them is **which feature is homogeneous?** A homogeneus feature is a feature that has similar distrubitions across all categories (something statiscally insignificant and cant help us classify)

To determine if a feature is homogeneous across categories, we can use a histogram to map the count of score ranges across all categories. The horizontal axis represent ranges of scores and the vertical axis represent the number of students who acquired that score

<table>
<tr>
<td>

![image](https://hackmd.io/_uploads/r1seZ9bmA.png)


</td>

<td>

Example of a homogeneous feature, notice the distributions are all similar across categories
    
</td>

</tr>

<tr>
<td>

![image](https://hackmd.io/_uploads/SJTb-9bmA.png)

</td>

<td>
    
Example of a non-homogeneous feature, notice the distributions are unique across categories

</td>

</tr>

</table>

Our next question is to **determine which features are similar?** Similar features should not be used in logistic regression because they contribute to the same weights anyway and will take up compute resources. A good way to determine this is to use a scatter plot against all features for all categories to find out how the values of the features relate to each other. The horizontal axis represent the values of a feature and the vertical axis represent the values of another feature.

<table>
<tr>
<td>

![image](https://hackmd.io/_uploads/H1oxQq-Q0.png)


</td>

<td>

Example of similar features, notice the features have a linear relationship, means both the features have the equivalent effect when classifying the categories
    
</td>

</tr>

<tr>
<td>

![image](https://hackmd.io/_uploads/H1K9XcWXC.png)

</td>

<td>
    
Example of unique features, notice the features form clusters, means both the features have the different effects when classifying the categories

</td>

</tr>

</table>

Now that we have established some baseline rules (no selecting homogeneous features and no selecting both similar features), we can start to select the feature we want our model to work with.


Here is an example of bad features to select (Arithmancy and potions):
![image](https://hackmd.io/_uploads/B1baVcbmC.png)

As you can see (from the horizontal axis - arithmancy), there is no clear classification of the features values. If we use this feature, then all values will have the same probability of being asssigned to any category. If we look at the vertical axis (potions), the same case applies as well.

Here is an example of a good feature being displayed:
![image](https://hackmd.io/_uploads/ByRPB5WXR.png)

The horizontal axis (Arithmancy) still remains a bad feature for the reasons above, but we can see some clustering now from the vertical axis (Charms). Right off the bat, we notice a high probability of getting classified as **Ravenclaw** when the Charms score is more than -240. Hence this feature will be used in the logistic regression model.

To be aware of more features like this, we have to repeat this mapping process for all the features, essentially generating something like a matrix.

![image](https://hackmd.io/_uploads/ByTcL5-mR.png)
> Pick your poison


# Training
Once you have picked your features, it is just a matter of applying the math formulas and execute **One vs Rest logistic regression with gradient descent**. A few things to note when implementing:
1. **Normalize** the data since the sigmoid function requires raising the value of $e$ to the power of your feature's values, which may go up to 1000 in this case. This will cause integer overflow if the feature values are not normalized. This goes without saying, you should also take this into account when making the predictions after the learning
2. **Handle NULL values**. Some rows in the training set will not have any feature values, we need to fill them up with the mean of that feature value so when they are put into account into logistic regression, those values will get registered as a 0.5 probability for all categories.
3. **Use batch / schohastic gradient descent**. Instead of running the GD algorithm for all rows, this approach optimizes the learning by randomly selecting rows to run GD on. This will improve speed(lesser computation time) as well as prevent overfitting (more generalized model). 

For the good features that have chosen, the regression line will look something like this:

![image](https://hackmd.io/_uploads/BJgHs5-QR.png)

For the bad features, you will get a flatter line:

![image](https://hackmd.io/_uploads/BJ9Us5b70.png)
