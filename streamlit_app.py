import streamlit as st
from collections import defaultdict
import itertools
import graphviz
import re

EPS = 'ε'

# ---------- Regex helpers ----------
def expand_plus(regex):
    pattern_group = re.compile(r'(\([^()]+\))\+')
    pattern_sym = re.compile(r'([A-Za-z0-9])\+')
    prev = None
    s = regex
    while prev != s:
        prev = s
        s = pattern_group.sub(r'\1.\1*', s)
        s = pattern_sym.sub(r'\1.\1*', s)
    return s

def insert_concat(regex):
    res = []
    prev = None
    symbols = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    for c in regex:
        if prev is not None:
            if (prev in symbols or prev in ")*?") and (c in symbols or c == '('):
                res.append('.')
        res.append(c)
        prev = c
    return ''.join(res)

def to_postfix(regex):
    prec = {'*':3, '?':3, '.':2, '|':1}
    output = []
    stack = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c.isalnum():
            output.append(c)
        elif c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec.get(c,0):
                output.append(stack.pop())
            stack.append(c)
        i += 1
    while stack:
        output.append(stack.pop())
    return ''.join(output)

# ---------- Thompson ----------
class Fragment:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def thompson_with_steps(postfix):
    transitions = defaultdict(list)
    counter = itertools.count()
    stack = []
    steps = []

    def snapshot(tok, new_transitions):
        copy_trans = {s:list(lst) for s,lst in transitions.items()}
        stack_repr = [f"[{frag.start}->{frag.accept}]" for frag in stack]
        steps.append({'tok': tok, 'stack': list(stack_repr), 'transitions': copy_trans, 'new': new_transitions})

    for tok in postfix:
        new_trans = []
        if tok.isalnum():
            s = next(counter); a = next(counter)
            transitions[s].append((tok,a))
            transitions.setdefault(a,[])
            new_trans.append((s,tok,a))
            stack.append(Fragment(s,a))
        elif tok == '.':
            f2 = stack.pop(); f1 = stack.pop()
            transitions[f1.accept].append((EPS,f2.start))
            new_trans.append((f1.accept,EPS,f2.start))
            stack.append(Fragment(f1.start,f2.accept))
        elif tok == '|':
            f2 = stack.pop(); f1 = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS,f1.start))
            transitions[s].append((EPS,f2.start))
            transitions[f1.accept].append((EPS,a))
            transitions[f2.accept].append((EPS,a))
            transitions.setdefault(s,[]); transitions.setdefault(a,[])
            new_trans += [(s,EPS,f1.start),(s,EPS,f2.start),(f1.accept,EPS,a),(f2.accept,EPS,a)]
            stack.append(Fragment(s,a))
        elif tok == '*':
            f = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS,f.start))
            transitions[s].append((EPS,a))
            transitions[f.accept].append((EPS,f.start))
            transitions[f.accept].append((EPS,a))
            transitions.setdefault(s,[]); transitions.setdefault(a,[])
            new_trans += [(s,EPS,f.start),(s,EPS,a),(f.accept,EPS,f.start),(f.accept,EPS,a)]
            stack.append(Fragment(s,a))
        elif tok == '?':
            f = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS,f.start))
            transitions[s].append((EPS,a))
            transitions[f.accept].append((EPS,a))
            transitions.setdefault(s,[]); transitions.setdefault(a,[])
            new_trans += [(s,EPS,f.start),(s,EPS,a),(f.accept,EPS,a)]
            stack.append(Fragment(s,a))
        snapshot(tok,new_trans)

    frag = stack.pop()
    final_nfa = {'start':frag.start,'accept':frag.accept,'transitions':dict(transitions)}
    return steps, final_nfa

# ---------- NFA → DFA ----------
def epsilon_closure(states,transitions):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for sym,d in transitions.get(s,[]):
            if sym==EPS and d not in closure:
                closure.add(d); stack.append(d)
    return closure

def move(states,symbol,transitions):
    result=set()
    for s in states:
        for sym,d in transitions.get(s,[]):
            if sym==symbol:
                result.add(d)
    return result

def nfa_to_dfa(nfa):
    trans = nfa['transitions']
    start = nfa['start']; accept = nfa['accept']
    symbols = sorted({sym for lst in trans.values() for sym,_ in lst if sym and sym!=EPS})

    start_set = frozenset(epsilon_closure({start},trans))
    unmarked=[start_set]
    dfa_states={start_set:0}
    dfa_trans={}; dfa_accepts=set()
    if accept in start_set: dfa_accepts.add(start_set)

    while unmarked:
        T=unmarked.pop()
        for sym in symbols:
            U = frozenset(epsilon_closure(move(T,sym,trans),trans))
            if not U: continue
            if U not in dfa_states:
                dfa_states[U]=len(dfa_states)
                unmarked.append(U)
                if accept in U: dfa_accepts.add(U)
            dfa_trans[(T,sym)] = U

    return {'states':list(dfa_states.keys()),'start':start_set,'accepts':dfa_accepts,'transitions':dfa_trans,'symbols':symbols}

# ---------- DFA Minimisation ----------
def minimize_dfa(dfa):
    states = list(dfa['states'])
    symbols = dfa['symbols']
    accepts = set(dfa['accepts'])
    non_accepts = set(states)-accepts
    P = [frozenset(accepts)] if accepts else []
    if non_accepts: P.append(frozenset(non_accepts))
    W = P.copy()
    trans = {s:{sym:dfa['transitions'].get((s,sym)) for sym in symbols} for s in states}

    while W:
        A=W.pop()
        for c in symbols:
            X={q for q in states if trans[q].get(c) in A}
            newP=[]
            for Y in P:
                inter = Y & X; diff = Y - X
                if inter and diff:
                    newP.append(frozenset(inter)); newP.append(frozenset(diff))
                    if Y in W: W.remove(Y); W.append(frozenset(inter)); W.append(frozenset(diff))
                    else: W.append(frozenset(inter) if len(inter)<=len(diff) else frozenset(diff))
                else: newP.append(Y)
            P=newP

    roman=["I","II","III","IV","V","VI","VII","VIII","IX","X"]
    state_names = {}; legend={}
    for i,block in enumerate(P):
        name = roman[i] if i<len(roman) else f"S{i}"
        state_names[block]=name
        legend[name]="{" + ",".join([str(s) for s in sorted(block)]) + "}"

    new_start = next(b for b in P if dfa['start'] in b)
    new_accepts = {b for b in P if any(s in dfa['accepts'] for s in b)}
    new_trans={}
    for src in P:
        for sym in symbols:
            dest=None
            for s in src:
                d=dfa['transitions'].get((s,sym))
                if d is not None:
                    dest=next(b for b in P if d in b); break
            if dest:
                key=(state_names[src],state_names[dest])
                if key in new_trans: new_trans[key].append(sym)
                else: new_trans[key]=[sym]

    return {'states':list(state_names.values()),'start':state_names[new_start],
            'accepts':{state_names[b] for b in new_accepts},'transitions':new_trans,
            'symbols':symbols,'legend':legend}

# ---------- Graph builder ----------
def build_nfa_graph(nfa,new_edges=None):
    g=graphviz.Digraph()
    g.attr(rankdir='LR',ranksep='1',nodesep='0.5')
    g.attr('node',shape='circle',fixedsize='true',width='1',height='1',fontsize='12')
    transitions = nfa['transitions']; start = nfa['start']; accept = nfa['accept']
    g.node('start',shape='point',width='0.1',height='0.1',fixedsize='true'); g.edge('start',f"n{start}")
    all_states=set(transitions.keys())
    for s,lst in transitions.items(): 
        for _,d in lst: all_states.add(d)
    for s in sorted(all_states,key=lambda x:int(str(x)) if str(x).isdigit() else str(x)):
        shape='doublecircle' if s==accept else 'circle'
        color='red' if s==start else 'black'
        g.node(f"n{s}",label=str(s),shape=shape,color=color,width='1',height='1',fixedsize='true')
    for s,lst in transitions.items():
        for sym,d in lst:
            label=EPS if sym==EPS or sym is None else str(sym)
            color="red" if new_edges and (s,sym,d) in new_edges else "black"
            g.edge(f"n{s}",f"n{d}",label=label,color=color)
    return g

def build_graph_minimized(dfa_min):
    g=graphviz.Digraph()
    g.attr(rankdir='LR',ranksep='1',nodesep='0.5')
    g.attr('node',shape='circle',fixedsize='true',width='1.2',height='1.2',fontsize='12')
    g.node('start',shape='point',width='0.1',height='0.1',fixedsize='true'); g.edge('start',dfa_min['start'])
    for s in dfa_min['states']:
        shape='doublecircle' if s in dfa_min['accepts'] else 'circle'
        g.node(s,label=s,shape=shape,width='1.2',height='1.2',fixedsize='true')
    for (src,dest),syms in dfa_min['transitions'].items():
        g.edge(src,dest,label=",".join(syms))
    return g

# ---------- Streamlit ----------
st.set_page_config(page_title="Algorithme de Thompson", layout="wide")
col_logo,col_title=st.columns([1,5])
with col_logo:
    try: st.image("logo.png",width=100)
    except: pass
with col_title:
    st.title("Algorithme de Thompson")
    st.caption("NFA → DFA → DFA minimisé (états abrégés + légende)")

regex = st.text_input("Expression régulière", value="(a|b)*ab(a|b)*")
colA,colB,colC=st.columns([1,1,1])
with colA: build_btn=st.button("Construire l'automate")
with colB: show_dfa=st.checkbox("Afficher DFA",value=True)
with colC: show_min=st.checkbox("Afficher DFA minimisé",value=True)

if 'steps' not in st.session_state: st.session_state.steps=[]
if 'final_nfa' not in st.session_state: st.session_state.final_nfa=None
if 'idx' not in st.session_state: st.session_state.idx=0

col1,col2=st.columns([1,2])
postfix_box=col1.empty(); info_box=col1.empty(); graph_box=col2.empty()
nav1,nav2=col1.columns([1,1])
prev_btn=nav1.button("← Étape précédente")
next_btn=nav2.button("Étape suivante")

if build_btn:
    try:
        regex_expanded=expand_plus(regex.strip())
        regex_concat=insert_concat(regex_expanded)
        postfix=to_postfix(regex_concat)
        steps, final_nfa=thompson_with_steps(postfix)
        st.session_state.steps=steps; st.session_state.final_nfa=final_nfa; st.session_state.idx=0
        st.success("NFA construit.")
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.session_state.steps=[]; st.session_state.final_nfa=None

if st.session_state.steps:
    if next_btn and st.session_state.idx<len(st.session_state.steps)-1: st.session_state.idx+=1
    if prev_btn and st.session_state.idx>0: st.session_state.idx-=1
    idx=st.session_state.idx; step=st.session_state.steps[idx]
    postfix_box.markdown(f"**Étape {idx+1}/{len(st.session_state.steps)} — Symbole traité :** {step['tok']}")
    info_box.write(f"Pile : {step['stack']}")
    temp_nfa={'start':st.session_state.final_nfa['start'],'accept':st.session_state.final_nfa['accept'],
              'transitions':step['transitions']}
    dot_nfa=build_nfa_graph(temp_nfa, step.get('new',[]))
    graph_box.graphviz_chart(dot_nfa.source)

    # DFA
    if show_dfa:
        dfa=nfa_to_dfa(st.session_state.final_nfa)
        # Multi-symbol DFA transitions
        dfa_trans_grouped={}
        for (src,sym),dest in dfa['transitions'].items():
            key=(src,dest)
            if key in dfa_trans_grouped: dfa_trans_grouped[key].append(sym)
            else: dfa_trans_grouped[key]=[sym]
        gdfa=graphviz.Digraph()
        gdfa.attr(rankdir='LR')
        gdfa.node('start',shape='point')
        gdfa.edge('start',str(dfa['start']))
        for s in dfa['states']:
            shape='doublecircle' if s in dfa['accepts'] else 'circle'
            gdfa.node(str(s),label=str(s),shape=shape)
        for (src,dest),syms in dfa_trans_grouped.items():
            gdfa.edge(str(src),str(dest),label=",".join(syms))
        st.subheader("DFA correspondant")
        st.graphviz_chart(gdfa.source)

    # DFA minimisé
    if show_min:
        dfa=nfa_to_dfa(st.session_state.final_nfa)
        min_dfa=minimize_dfa(dfa)
        st.subheader("DFA minimisé (états abrégés)")
        gmin=build_graph_minimized(min_dfa)
        st.graphviz_chart(gmin.source)
        st.markdown("\n".join([f"- **{k}** = `{v}`" for k,v in min_dfa['legend'].items()]))
else:
    st.info("Entrez une expression régulière et cliquez sur *Construire l'automate*.")
