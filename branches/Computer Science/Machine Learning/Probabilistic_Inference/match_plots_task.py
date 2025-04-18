#!/usr/bin/env python3
"""
Generate a "Probabilistic Inference" task involving priors, a simple Gaussian
likelihood, and posterior distributions. This script:
  1) Defines a normal likelihood with given observations.
  2) Defines 4 different prior distributions over theta.
  3) Plots the priors for identification.
  4) Prints LaTeX code for a problem statement referencing these priors.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def gaussian(x, mu, sigma):
    """Compute value of Gaussian PDF with mean=mu and std=sigma at x."""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def laplace(x, mu, b):
    """Compute Laplace PDF with location=mu and scale=b at x."""
    return (1.0 / (2*b)) * np.exp(-np.abs(x - mu) / b)

def uniform(x, a, b):
    """Compute Uniform PDF on [a, b] evaluated at x."""
    # Returns 1/(b-a) if x in [a,b], else 0
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)

def mixture_of_gaussians(x, params):
    """
    Compute a mixture of Gaussians PDF for each x.
    params = [(w1, mu1, sigma1), (w2, mu2, sigma2), ...]
    Each weight wi should sum to 1.
    """
    pdf_vals = np.zeros_like(x, dtype=float)
    for (w, mu, sigma) in params:
        pdf_vals += w * gaussian(x, mu, sigma)
    return pdf_vals

def generate_problem_statement():
    # 1. Define observation(s)
    #    For example, three synthetic data points from some normal model:
    observations = np.array([10., 3., 5.])

    # 2. Define a simple Gaussian likelihood: p(x | theta) = Normal(x | theta, sigma=1)

    # 3. Define range for plotting
    theta_vals = np.linspace(-2, 12, 1000)

    # 4. Define priors p(theta)
    #    a) A normal prior with mean=-1 and std=1.5 (for example)
    prior_a = gaussian(theta_vals, mu=-1, sigma=1.5)

    #    b) A uniform prior between 4 and 8
    prior_b = uniform(theta_vals, a=4, b=8)

    #    c) A Laplace prior with location=2, scale=1.2
    prior_c = laplace(theta_vals, mu=2, b=1.2)

    #    d) A mixture of two Gaussians
    #       Suppose w1=0.6, mu1=6, sigma1=0.5; w2=0.4, mu2=9, sigma2=1.0
    mixture_params = [(0.6, 6, 0.5), (0.4, 9, 1.0)]
    prior_d = mixture_of_gaussians(theta_vals, mixture_params)

    # Normalize each prior so it visually looks like a PDF (not mandatory, but helpful for plotting)
    prior_a /= np.trapz(prior_a, theta_vals)
    prior_b /= np.trapz(prior_b, theta_vals)
    prior_c /= np.trapz(prior_c, theta_vals)
    prior_d /= np.trapz(prior_d, theta_vals)

    # 5. Plot the four priors on the same figure
    plt.figure(figsize=(8,5))
    plt.plot(theta_vals, prior_a, label="Prior A")
    plt.plot(theta_vals, prior_b, label="Prior B")
    plt.plot(theta_vals, prior_c, label="Prior C")
    plt.plot(theta_vals, prior_d, label="Prior D")
    plt.title("Priors over Theta")
    plt.xlabel("Theta")
    plt.ylabel("p(Theta)")
    plt.legend()
    
    # Save figure
    fig_name = "Machine Learning/03_Probabilistic_Inference/data/priors_comparison.png"
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Produce a minimal LaTeX "problem" snippet
    latex_doc = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \usepackage{graphicx}
        \begin{document}

        \section*{Probabilistic Inference Task}

        Consider the probabilistic model
        \[
            p(x \mid \theta) = \mathcal{N}(x \mid \theta, 1),
        \]
        and a set of observations 
        \[
            \{x_1, x_2, x_3\} = \{10, 3, 5\}.
        \]

        \noindent
        We have four candidate prior distributions $p(\theta)$ shown in the figure below:

        \begin{center}
        \includegraphics[width=0.8\textwidth]{REPLACE_WITH_FIGURE_NAME}
        \end{center}

        \begin{itemize}
        \item[a)] $p(\theta) = \mathcal{N}(\theta \mid -1, \,1.5^2)$
        \item[b)] $p(\theta) = \mathrm{Uniform}(4,\,8)$
        \item[c)] $p(\theta) \propto \exp\bigl(-|\theta - 2|/1.2\bigr)$
        \item[d)] $p(\theta) = 0.6\,\mathcal{N}(\theta \mid 6,\,0.5^2) \;+\; 0.4\,\mathcal{N}(\theta \mid 9,\,1.0^2)$
        \end{itemize}

        Match each of the labels (A--D) to the corresponding colored curves in the above figure.
        Then, using the likelihood $p(\{10,3,5\} \mid \theta)$ and your chosen prior, derive 
        and write down the functional form of the posterior $p(\theta \mid \{10,3,5\})$ (unnormalized).
        Finally, discuss briefly how you would find the maximum a posteriori (MAP) estimate 
        and/or the posterior mean.

        \end{document}
    """

    # Replace the placeholder with the actual figure name
    latex_doc = latex_doc.replace("REPLACE_WITH_FIGURE_NAME", fig_name)

    # 7. Write the LaTeX code to an output file (optional). For demonstration, we'll just print it:
    out_filename = "problem_statement.tex"
    with open(out_filename, "w") as f:
        f.write(latex_doc)
    print(f"[INFO] LaTeX problem statement written to '{out_filename}'.")
    
    # Also display to screen
    print("===================================== LaTeX Problem Statement =====================================")
    print(latex_doc)
    print("====================================================================================================")

def main():
    # Run the generation
    generate_problem_statement()

if __name__ == "__main__":
    main()
