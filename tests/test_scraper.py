import unittest

from bs4 import BeautifulSoup

from src.scraper import _clean_spec_div, _clean_statement_html


class StatementScraperTests(unittest.TestCase):
    def test_tex_math_is_converted_to_readable_notation(self):
        markup = r"""
        <div class="problem-statement">
            <p>Limits: $$$1 \le n \le 5\cdot 10^{5}$$$.</p>
            <p>Values $$$a_i, l_i, r_i, x_i^2$$$.</p>
            <p>Greek $$$\alpha + \beta$$$; fraction $$$\frac{n+1}{2}$$$.</p>
            <p>Operators $$$x \times y \neq 0$$$.</p>
        </div>
        """
        statement = _clean_statement_html(BeautifulSoup(markup, "lxml").div)

        self.assertIn("1 ≤ n ≤ 5·10^5", statement)
        self.assertIn("a_i, l_i, r_i, x_i^2", statement)
        self.assertIn("α + β", statement)
        self.assertIn("(n+1)/2", statement)
        self.assertIn("x × y ≠ 0", statement)

    def test_html_sub_sup_entities_and_mathjax_attributes(self):
        markup = """
        <div class="input-specification">
            <p>1 &le; <i>n</i> &le; 5&middot;10<sup>5</sup>.</p>
            <p><i>a</i><sub>i</sub>, <i>l</i><sub>i</sub>, <i>r</i><sub>i</sub>.</p>
            <p><span class="MathJax" alt="x_i^2 + &alpha; = &frac12;"></span></p>
            <p><span data-tex="\\frac{a+b}{c}"></span></p>
        </div>
        """
        result = _clean_spec_div(BeautifulSoup(markup, "lxml").div)

        self.assertIn("1 ≤ n ≤ 5·10^5", result)
        self.assertIn("a_i, l_i, r_i", result)
        self.assertIn("x_i^2 + α = ½", result)
        self.assertIn("(a+b)/c", result)

    def test_mathjax_preview_and_rendered_image_are_not_duplicated(self):
        markup = r"""
        <div class="problem-statement"><p>
            <span class="MathJax_Preview">\(x_i^2 + \alpha\)</span>
            <span class="MathJax"><img alt="x_i^2 + \alpha" /></span>
        </p></div>
        """
        result = _clean_statement_html(BeautifulSoup(markup, "lxml").div)

        self.assertEqual(result, "x_i^2 + α")

    def test_sample_input_output_and_statement_code_are_preserved(self):
        markup = """
        <div class="problem-statement">
            <p>Example code follows.</p>
            <pre>int main() {\n    return 0;\n}</pre>
            <div class="sample-tests"><div class="sample-test">
                <div class="input"><div class="title">Input</div><pre>2\n1 2</pre></div>
                <div class="output"><div class="title">Output</div><pre>3</pre></div>
            </div></div>
        </div>
        """
        result = _clean_statement_html(BeautifulSoup(markup, "lxml").div)

        self.assertIn("```\nint main() {\n    return 0;\n}\n```", result)
        self.assertIn("Input:\n```\n2\n1 2\n```", result)
        self.assertIn("Output:\n```\n3\n```", result)


if __name__ == "__main__":
    unittest.main()
