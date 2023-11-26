import click

from tetrimino.tetrimino import *


class GridSizeType(click.ParamType):
    name = "grid size"

    def convert(self, value, param, ctx) -> Dimensions:
        if not isinstance(value, str):
            return value

        err_string = f"Invalid argument for {param}: {value} - should be of format '{{rows}}x{{cols}}'"

        dimensions = value.split("x")
        if len(dimensions) != 2:
            self.fail(err_string)

        try:
            row, col = map(int, dimensions)
            return row, col

        except ValueError:
            self.fail("Could not convert rows or cols to integers.")


@click.command()
@click.option("-s/-g", "--solve/--generate", default=False, help="Number of greetings.")
@click.option(
    "-v/-q",
    "--verbose/--quiet",
    default=False,
    help="Whether to print nodes as we search",
)
@click.argument(
    "size",
    type=GridSizeType(),
)
def tiler(size: Dimensions, solve: bool, verbose: bool):
    method = solve_constraints if solve else generate_solution
    method_text = "Solving" if solve else "Generating"
    click.echo(
        "{} grid with dimensions: {}".format(
            click.style(method_text, bold=True),
            click.style(size, bold=True, fg="green"),
        )
    )
    grid = create_empty_grid(size)
    solution = method(grid, [t.value for t in Tetriminos], verbose)
    click.echo("Finished!")
    print()
    print(solution)


def main():
    tiler()


if __name__ == "__main__":
    main()
