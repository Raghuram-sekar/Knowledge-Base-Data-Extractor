

# Set tokenizers parallelism to false to avoid fork warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

if __name__ == "__main__":
    from pipeline.parsing import main
    sys.exit(main())
