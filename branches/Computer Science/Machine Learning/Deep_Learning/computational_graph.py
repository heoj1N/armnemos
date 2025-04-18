# save this script as e.g. draw_ops.py, then run: python draw_ops.py
# It will create a series of .svg files, each depicting one operation.

from graphviz import Digraph

def draw_negative():
    dot = Digraph('NegativeOperation', format='png')
    dot.node('x',    'x',          shape='ellipse')
    dot.node('op',   'Negative',   shape='box')
    dot.node('out',  'out = -x',   shape='ellipse')
    
    # Forward pass
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward pass
    # d_out flows into op; op passes gradients to x
    dot.edge('op', 'x',   label='d_x = -d_out', color='red')
    return dot

def draw_add():
    dot = Digraph('AddOperation', format='png')
    dot.node('x',   'x',          shape='ellipse')
    dot.node('y',   'y',          shape='ellipse')
    dot.node('op',  'Add',        shape='box')
    dot.node('out', 'out = x+y',  shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out', color='red')
    dot.edge('op', 'y', label='d_y = d_out', color='red')
    return dot

def draw_subtract():
    dot = Digraph('SubtractOperation', format='png')
    dot.node('x',   'x',           shape='ellipse')
    dot.node('y',   'y',           shape='ellipse')
    dot.node('op',  'Subtract',    shape='box')
    dot.node('out', 'out = x-y',   shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = +d_out', color='red')
    dot.edge('op', 'y', label='d_y = -d_out', color='red')
    return dot

def draw_multiply():
    dot = Digraph('MultiplyOperation', format='png')
    dot.node('x',   'x',            shape='ellipse')
    dot.node('y',   'y',            shape='ellipse')
    dot.node('op',  'Multiply',     shape='box')
    dot.node('out', 'out = x*y',    shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out*y', color='red')
    dot.edge('op', 'y', label='d_y = d_out*x', color='red')
    return dot

def draw_divide():
    dot = Digraph('DivideOperation', format='png')
    dot.node('x',   'x',            shape='ellipse')
    dot.node('y',   'y',            shape='ellipse')
    dot.node('op',  'Divide',       shape='box')
    dot.node('out', 'out = x/y',    shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out*(1/y)', color='red')
    dot.edge('op', 'y', label='d_y = d_out*(-x/y^2)', color='red')
    return dot

def draw_sum():
    dot = Digraph('SumOperation', format='png')
    dot.node('x',   'x (array)',    shape='ellipse')
    dot.node('op',  'Sum',          shape='box')
    dot.node('out', 'out = Σ x_i',  shape='ellipse')
    
    # Forward
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out*ones_like(x)', color='red')
    return dot

def draw_dotproduct():
    dot = Digraph('DotProductOperation', format='png')
    dot.node('x',   'x (vector)',      shape='ellipse')
    dot.node('y',   'y (vector)',      shape='ellipse')
    dot.node('op',  'DotProduct',      shape='box')
    dot.node('out', 'out = x⋅y',       shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out * y', color='red')
    dot.edge('op', 'y', label='d_y = d_out * x', color='red')
    return dot

def draw_exp():
    dot = Digraph('ExpOperation', format='png')
    dot.node('x',   'x',         shape='ellipse')
    dot.node('op',  'Exp',       shape='box')
    dot.node('out', 'out=e^x',   shape='ellipse')
    
    # Forward
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out * e^x', color='red')
    return dot

def draw_log():
    dot = Digraph('LogOperation', format='png')
    dot.node('x',   'x',          shape='ellipse')
    dot.node('op',  'Log',        shape='box')
    dot.node('out', 'out=log(x)', shape='ellipse')
    
    # Forward
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out*(1/x)', color='red')
    return dot

def draw_power():
    dot = Digraph('PowerOperation', format='png')
    dot.node('x',   'x',           shape='ellipse')
    dot.node('op',  'Power(p)',    shape='box')
    dot.node('out', 'out = x^p',   shape='ellipse')
    
    # Forward
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x = d_out * p*x^(p-1)', color='red')
    return dot

def draw_max():
    dot = Digraph('MaxOperation', format='png')
    dot.node('x',   'x',            shape='ellipse')
    dot.node('y',   'y',            shape='ellipse')
    dot.node('op',  'Max',          shape='box')
    dot.node('out', 'out=max(x,y)', shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x=d_out if x>=y else 0', color='red')
    dot.edge('op', 'y', label='d_y=d_out if y> x else 0', color='red')
    return dot

def draw_min():
    dot = Digraph('MinOperation', format='png')
    dot.node('x',   'x',            shape='ellipse')
    dot.node('y',   'y',            shape='ellipse')
    dot.node('op',  'Min',          shape='box')
    dot.node('out', 'out=min(x,y)', shape='ellipse')
    
    # Forward
    dot.edge('x', 'op', label='Forward', color='blue')
    dot.edge('y', 'op', label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    dot.edge('op', 'x', label='d_x=d_out if x<=y else 0', color='red')
    dot.edge('op', 'y', label='d_y=d_out if y< x else 0', color='red')
    return dot

def draw_clip():
    dot = Digraph('ClipOperation', format='png')
    dot.node('x',   'x',                           shape='ellipse')
    dot.node('op',  'Clip(min_val, max_val)',       shape='box')
    dot.node('out', 'out=clip(x, min_val, max_val)',shape='ellipse')
    
    # Forward
    dot.edge('x', 'op',   label='Forward', color='blue')
    dot.edge('op', 'out', label='Forward', color='blue')
    
    # Backward
    # "pass_mask" means derivative passes through only if x is in [min_val, max_val]
    dot.edge('op', 'x', label='d_x = d_out * pass_mask', color='red')
    return dot


if __name__ == '__main__':
    # Create and render each operation diagram:
    draw_negative().render('data/Negative', cleanup=True)
    draw_add().render('data/Add', cleanup=True)
    draw_subtract().render('data/Subtract', cleanup=True)
    draw_multiply().render('data/Multiply', cleanup=True)
    draw_divide().render('data/Divide', cleanup=True)
    draw_sum().render('data/Sum', cleanup=True)
    draw_dotproduct().render('data/DotProduct', cleanup=True)
    draw_exp().render('data/Exp', cleanup=True)
    draw_log().render('data/Log', cleanup=True)
    draw_power().render('data/Power', cleanup=True)
    draw_max().render('data/Max', cleanup=True)
    draw_min().render('data/Min', cleanup=True)
    draw_clip().render('data/Clip', cleanup=True)

    print("All diagrams have been generated as .png files in the data directory.")
