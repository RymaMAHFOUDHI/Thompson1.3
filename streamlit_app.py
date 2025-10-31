import streamlit as st
from collections import defaultdict
import itertools
import graphviz
import re

EPS = 'ε'

# ---------- Fonctions regex ----------
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
            if not stack:
                raise ValueError("Parenthèse fermante sans ouvrante")
            stack.pop()
        else:
            while stack and stack[-1] != '(' and prec.get(stack[-1],0) >= prec.get(c,0):
                output.append(stack.pop())
            stack.append(c)
        i += 1
    while stack:
        op = stack.pop()
        if op in '()':
            raise ValueError("Parenthèses mal appariées")
        output.append(op)
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
        copy_trans = {s: list(lst) for s, lst in transitions.items()}
        stack_repr = [f"[{frag.start}->{frag.accept}]" for frag in stack]
        steps.append({'tok': tok, 'stack': list(stack_repr), 'transitions': copy_trans, 'new': new_transitions})

    for tok in postfix:
        new_trans = []
        if tok.isalnum():
            s = next(counter); a = next(counter)
            transitions[s].append((tok, a))
            transitions.setdefault(a, [])
            new_trans.append((s, tok, a))
            stack.append(Fragment(s, a))
        elif tok == '.':
            f2 = stack.pop(); f1 = stack.pop()
            transitions[f1.accept].append((EPS, f2.start))
            new_trans.append((f1.accept, EPS, f2.start))
            stack.append(Fragment(f1.start, f2.accept))
        elif tok == '|':
            f2 = stack.pop(); f1 = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS, f1.start))
            transitions[s].append((EPS, f2.start))
            transitions[f1.accept].append((EPS, a))
            transitions[f2.accept].append((EPS, a))
            transitions.setdefault(s, [])
            transitions.setdefault(a, [])
            new_trans += [(s, EPS, f1.start), (s, EPS, f2.start), (f1.accept, EPS, a), (f2.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '*':
            f = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS, f.start))
            transitions[s].append((EPS, a))
            transitions[f.accept].append((EPS, f.start))
            transitions[f.accept].append((EPS, a))
            transitions.setdefault(s, [])
            transitions.setdefault(a, [])
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, f.start), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        elif tok == '?':
            f = stack.pop()
            s = next(counter); a = next(counter)
            transitions[s].append((EPS, f.start))
            transitions[s].append((EPS, a))
            transitions[f.accept].append((EPS, a))
            transitions.setdefault(s, [])
            transitions.setdefault(a, [])
            new_trans += [(s, EPS, f.start), (s, EPS, a), (f.accept, EPS, a)]
            stack.append(Fragment(s, a))
        snapshot(tok, new_trans)

    frag = stack.pop()
    final_nfa = {'start': frag.start, 'accept': frag.accept, 'transitions': dict(transitions)}
    return steps, final_nfa

# ---------- NFA → DFA ----------
def epsilon_closure(states, transitions):
    closure = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for sym, d in transitions.get(s, []):
            if sym == EPS and d not in closure:
                closure.add(d)
                stack.append(d)
    return closure

def move(states, symbol, transitions):
    result = set()
    for s in states:
        for sym, d in transitions.get(s, []):
            if sym == symbol:
                result.add(d)
    return result

def nfa_to_dfa(nfa):
    transitions = nfa['transitions']
    start = nfa['start']
    accept = nfa['accept']
    symbols = sorted(set(sym for lst in transitions.values() for sym,_ in lst if sym and sym != EPS))

    start_set = frozenset(epsilon_closure({start}, transitions))
    unmarked = [start_set]
    dfa_states = {start_set: 0}
    dfa_trans = {}
    dfa_accepts = set()
    if accept in start_set:
        dfa_accepts.add(start_set)

    sink = frozenset({'sink'})
    sink_needed = False
    while unmarked:
        T = unmarked.pop()
        for sym in symbols:
            Uset = epsilon_closure(move(T, sym, transitions), transitions)
            U = frozenset(Uset)
            if not U:
                sink_needed = True
                dfa_states.setdefault(sink, len(dfa_states))
                dfa_trans[(T, sym)] = sink
                continue
            if U not in dfa_states:
                dfa_states[U] = len(dfa_states)
                unmarked.append(U)
                if accept in U:
                    dfa_accepts.add(U)
            dfa_trans[(T, sym)] = U

    if sink_needed:
        for sym in symbols:
            dfa_trans[(sink, sym)] = sink

    return {'states': list(dfa_states.keys()), 'start': start_set, 'accepts': dfa_accepts, 'transitions': dfa_trans, 'symbols': symbols}

# ---------- Graphiques NFA / DFA multisymbole ----------
def build_graph_multisym(nfa_or_dfa, start_key='start', accept_key='accept', transitions_key='transitions', multi_symbol=True, highlight_edges=None):
    g = graphviz.Digraph()
    g.attr(rankdir='LR', ranksep='1', nodesep='0.5')
    g.attr('node', shape='circle', fixedsize='true', width='1.2', height='1.2', fontsize='12')

    start = nfa_or_dfa[start_key]
    accept = nfa_or_dfa[accept_key]
    transitions = nfa_or_dfa[transitions_key]

    # Point initial
    g.node('start', shape='point', width='0.1', height='0.1', fixedsize='true')
    g.edge('start', str(start), color="red")

    # États
    all_states = set()
    if start_key=='start':
        all_states.update(transitions.keys())
        for lst in transitions.values():
            for _, d in lst if isinstance(lst[0], tuple) else [(None,None)]:
                all_states.add(d)
    for s in all_states:
        shape = 'doublecircle' if s == accept or (isinstance(accept,set) and s in accept) else 'circle'
        g.node(str(s), label=str(s), shape=shape, width='1.2', height='1.2', fixedsize='true')

    # Transitions regroupées si nécessaire
    if multi_symbol:
        trans_dict = defaultdict(list)
        # NFA format : dict s -> list of (sym,d)
        if isinstance(transitions, dict) and all(isinstance(v,list) for v in transitions.values()):
            for s,lst in transitions.items():
                for sym,d in lst:
                    trans_dict[(s,d)].append(EPS if sym==EPS else str(sym))
        # DFA format : dict (src,sym) -> dst
        elif isinstance(transitions, dict) and all(isinstance(k,tuple) for k in transitions.keys()):
            for (src,sym),dst in transitions.items():
                trans_dict[(src,dst)].append(str(sym))
        for (src,dst),syms in trans_dict.items():
            color = "red" if highlight_edges and (src, dst) in highlight_edges else "black"
            g.edge(str(src), str(dst), label=",".join(sorted(syms)), color=color)
    else:
        # trans normale
        if isinstance(transitions, dict) and all(isinstance(v,list) for v in transitions.values()):
            for s,lst in transitions.items():
                for sym,d in lst:
                    g.edge(str(s), str(d), label=EPS if sym==EPS else str(sym))
        elif isinstance(transitions, dict) and all(isinstance(k,tuple) for k in transitions.keys()):
            for (src,sym),dst in transitions.items():
                g.edge(str(src), str(dst), label=str(sym))

    return g

# ---------- Streamlit ----------
st.set_page_config(page_title="Algorithme de Thompson (abrégé)", layout="wide")
col_logo, col_title = st.columns([1,5])
with col_logo:
    try:
        st.image("logo.png", width=100)
    except: pass
with col_title:
    st.title("Algorithme de Thompson")
    st.caption("Construction NFA → DFA (multi-symbole)")

st.header("Entrée")
regex = st.text_input("Expression régulière", value="(a|b)*ab(a|b)*")

colA, colB, colC = st.columns([1,1,1])
with colA: build = st.button("Construire l'automate")
with colB: show_dfa = st.checkbox("Afficher DFA", value=True)
with colC: show_min = st.checkbox("Afficher DFA minimisé", value=False)

st.divider()
if 'steps' not in st.session_state: st.session_state.steps=[]
if 'final_nfa' not in st.session_state: st.session_state.final_nfa=None
if 'idx' not in st.session_state: st.session_state.idx=0

col1, col2 = st.columns([1,2])
postfix_box = col1.empty()
info_box = col1.empty()
graph_box = col2.empty()
nav1, nav2 = col1.columns([1,1])
prev_btn = nav1.button("← Étape précédente")
next_btn = nav2.button("Étape suivante")

# ---------- Construction ----------
if build:
    try:
        regex_expanded = expand_plus(regex.strip())
        regex2 = insert_concat(regex_expanded)
        postfix = to_postfix(regex2)
        steps, final_nfa = thompson_with_steps(postfix)
        st.session_state.steps=steps
        st.session_state.final_nfa=final_nfa
        st.session_state.idx=0
        st.success("Automate NFA construit.")
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.session_state.steps=[]
        st.session_state.final_nfa=None

# ---------- Affichage étapes NFA ----------
if st.session_state.steps:
    if next_btn and st.session_state.idx<len(st.session_state.steps)-1: st.session_state.idx+=1
    if prev_btn and st.session_state.idx>0: st.session_state.idx-=1
    idx = st.session_state.idx
    step = st.session_state.steps[idx]
    postfix_box.markdown(f"**Étape {idx+1}/{len(st.session_state.steps)} — Symbole traité :** {step['tok']}")
    info_box.write(f"Pile : {step['stack']}")
    dot_nfa = build_graph_multisym({'start': st.session_state.final_nfa['start'],
                                    'accept': st.session_state.final_nfa['accept'],
                                    'transitions': step['transitions']})
    graph_box.graphviz_chart(dot_nfa.source)

# ---------- Affichage DFA ----------
if st.session_state.final_nfa:
    dfa = nfa_to_dfa(st.session_state.final_nfa)
    if show_dfa:
        st.subheader("DFA correspondant")
        g_dfa = build_graph_multisym(dfa, start_key='start', accept_key='accept', transitions_key='transitions')
        st.graphviz_chart(g_dfa.source)

    if show_min:
        # minimisation DFA (version simple)
        from copy import deepcopy
        def minimize_dfa(dfa):
            states=list(dfa['states']);accepts=set(dfa['accepts']);non_accepts=set(states)-accepts
            P=[frozenset(accepts) if accepts else None, frozenset(non_accepts) if non_accepts else None]
            P=[p for p in P if p];W=P.copy()
            symbols=dfa['symbols']
            trans={s:{sym:dfa['transitions'].get((s,sym)) for sym in symbols} for s in states}
            while W:
                A=W.pop()
                for c in symbols:
                    X={q for q in states if trans[q].get(c) in A}
                    newP=[]
                    for Y in P:
                        inter=Y & X; diff=Y-X
                        if inter and diff:
                            newP.append(frozenset(inter));newP.append(frozenset(diff))
                            if Y in W: W.remove(Y); W.append(frozenset(inter));W.append(frozenset(diff))
                            else: W.append(frozenset(inter) if len(inter)<=len(diff) else frozenset(diff))
                        else: newP.append(Y)
                    P=newP
            new_states=P
            new_start=next(b for b in new_states if dfa['start'] in b)
            new_accepts={b for b in new_states if any(s in dfa['accepts'] for s in b)}
            new_trans={}
            for b in new_states:
                rep=next(iter(b))
                for sym in symbols:
                    dst=dfa['transitions'].get((rep,sym))
                    if dst is None: continue
                    dst_blk=next(bl for bl in new_states if dst in bl)
                    new_trans[(b,sym)]=dst_blk
            return {'states':new_states,'start':new_start,'accepts':new_accepts,'transitions':new_trans,'symbols':symbols}
        min_dfa=minimize_dfa(dfa)
        st.subheader("DFA minimisé")
        g_min=build_graph_multisym(min_dfa, start_key='start', accept_key='accept', transitions_key='transitions')
        st.graphviz_chart(g_min.source)
