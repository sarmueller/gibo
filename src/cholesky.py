import torch


def one_step_cholesky(
    top_left: torch.Tensor, K_Xθ: torch.Tensor, K_θθ: torch.Tensor, A_inv: torch.Tensor
) -> torch.Tensor:
    '''Update the Cholesky factor when the matrix is extended.

    Note: See thesis appendix A.2 for notation of args and further information.

    Args:
        top_left: Cholesky factor L11 of old matrix A11.
        K_Xθ: Upper right bock matrix A12 of new matrix A.
        K_θθ: Lower right block matrix A22 of new matrix A.
        A_inv: Inverse of old matrix A11.

    Returns:
        New cholesky factor S of new matrix A.
    '''
    # Solve with A \ b: A @ x = b, x = A^(-1) @ b,
    # top_right = L11^T \ A12 = L11^T  \ K_Xθ, top_right = (L11^T)^(-1) @ K_Xθ,
    # Use: (L11^(-1))^T = L11 @ A11^(-1).
    # Hint: could also be solved with torch.cholesky_solve (in theory faster).
    top_right = top_left @ (A_inv @ K_Xθ)
    bot_left = torch.zeros_like(top_right).transpose(-1, -2)
    bot_right = torch.cholesky(
        K_θθ - top_right.transpose(-1, -2) @ top_right, upper=True
    )
    return torch.cat(
        [
            torch.cat([top_left, top_right], dim=-1),
            torch.cat([bot_left, bot_right], dim=-1),
        ],
        dim=-2,
    )
