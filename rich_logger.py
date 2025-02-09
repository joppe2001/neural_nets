from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.columns import Columns
from typing import Dict, List, Union
import numpy as np
import logging


class RichLogger:
    def __init__(self, console: Console = None):
        self.console = console or Console()
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console)]
        )
        self.log = logging.getLogger("rich")

    def create_table(self, data: Union[np.ndarray, Dict], title: str = "") -> Table:
        """Create a table from data"""
        table = Table(title=title, expand=False)  # expand=False prevents table from stretching

        if isinstance(data, np.ndarray):
            for i in range(data.shape[1] if len(data.shape) > 1 else 1):
                table.add_column(f"Col {i + 1}")
            if len(data.shape) == 1:
                table.add_row(*[f"{x:.4f}" if isinstance(x, float) else str(x) for x in data])
            else:
                for row in data:
                    table.add_row(*[f"{x:.4f}" if isinstance(x, float) else str(x) for x in row])
        else:  # Dictionary
            table.add_column("Key")
            table.add_column("Value")
            for key, value in data.items():
                table.add_row(str(key), str(value))

        return table

    def print_items(self, items: List[tuple], side_by_side: bool = False) -> None:
        """Print multiple items with their titles and styles
        items: List of tuples (data, title, border_style)"""

        panels = []
        for data, title, style in items:
            table = self.create_table(data, title)
            panels.append(Panel(table, title=title, border_style=style))

        if side_by_side and len(panels) <= 4:  # Only show side by side if 2 or fewer panels
            self.console.print(Columns(panels, expand=False))
        else:
            for panel in panels:
                self.console.print(panel)

    def print_array(self, data: np.ndarray, title: str = "", border_style: str = "blue",
                    side_by_side: bool = False) -> None:
        """Pretty print numpy arrays"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.print_items([(data, title, border_style)], side_by_side)

    def print_dict(self, data: Dict, title: str = "", border_style: str = "green",
                   side_by_side: bool = False) -> None:
        """Pretty print dictionaries"""
        self.print_items([(data, title, border_style)], side_by_side)

    def print_metrics(self, metrics: Dict[str, float], title: str = "Metrics",
                      border_style: str = "yellow", side_by_side: bool = False) -> None:
        """Pretty print training metrics"""
        self.print_items([(metrics, title, border_style)], side_by_side)

    def section(self, text: str, style: str = "bold blue") -> None:
        """Print a section header"""
        self.console.print(f"\n[{style}]{'=' * 20} {text} {'=' * 20}[/{style}]\n")

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def error(self, msg: str) -> None:
        self.log.error(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)
