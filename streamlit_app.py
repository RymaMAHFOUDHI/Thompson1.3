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

# ---------- DFA minimisation ----------
def minimize_dfa(dfa):
    states = list(dfa['states'])
    symbols = dfa['symbols']
    accepts = set(dfa['accepts'])
    non_accepts = set(states) - accepts

    P = []
    if accepts: P.append(frozenset(accepts))
    if non_accepts: P.append(frozenset(non_accepts))
    W = [p for p in P]

    trans = {s:{} for s in states}
    for s in states:
        for sym in symbols:
            trans[s][sym] = dfa['transitions'].get((s,sym), None)

    while W:
        A = W.pop()
        for c in symbols:
            X = set(q for q in states if trans[q].get(c) in A)
            newP = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    newP.append(frozenset(inter))
                    newP.append(frozenset(diff))
                    if Y in W:
                        W.remove(Y)
                        W.append(frozenset(inter))
                        W.append(frozenset(diff))
                    else:
                        W.append(frozenset(inter) if len(inter)<=len(diff) else frozenset(diff))
                else:
                    newP.append(Y)
            P = newP

    # Renommer I, II, III
    mapping = {blk: roman(i+1) for i, blk in enumerate(P)}
    new_trans = {}
    for blk in P:
        repr_state = next(iter(blk))
        for sym in symbols:
            dest = dfa['transitions'].get((repr_state,sym))
            if dest:
                for dst_blk in P:
                    if dest in dst_blk:
                        new_trans[(blk,sym)] = dst_blk
                        break
    new_start = next(blk for blk in P if dfa['start'] in blk)
    new_accepts = set(blk for blk in P if any(s in dfa['accepts'] for s in blk))

    return {'states': P, 'start': new_start, 'accepts': new_accepts, 'transitions': new_trans, 'symbols': symbols, 'mapping': mapping}

def roman(n):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),
            (40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    res=''
    for v,s in vals:
        while n>=v: res+=s;n-=v
    return res

# ---------- Graphe multi-symbole ----------
def build_graph_multisym(nfa_or_dfa, start_key='start', accept_key=None, transitions_key='transitions', mapping=None):
    g = graphviz.Digraph()
    g.attr(rankdir='LR', ranksep='1', nodesep='0.5')
    g.attr('node', shape='circle', fixedsize='true', width='1.2', height='1.2', fontsize='12')

    start = nfa_or_dfa[start_key]
    if accept_key is None:
        if 'accept' in nfa_or_dfa:
            accept = nfa_or_dfa['accept']
        elif 'accepts' in nfa_or_dfa:
            accept = nfa_or_dfa['accepts']
        else:
            accept = set()
    else:
        accept = nfa_or_dfa.get(accept_key)
    
    transitions = nfa_or_dfa[transitions_key]

    g.node('start', shape='point', width='0.1', height='0.1', fixedsize='true')
    start_label = mapping[start] if mapping else str(start)
    g.edge('start', start_label, color='red')

    all_states = set(transitions.keys())
    for lst in transitions.values():
        if isinstance(lst,list):
            for pair in lst:
                if isinstance(pair, tuple) and len(pair)==2:
                    _, d = pair
                    all_states.add(d)
                else:
                    all_states.add(pair)
        else:
            all_states.add(lst)
    if isinstance(accept,(set,frozenset)):
        all_states.update(accept)
    else:
        all_states.add(accept)

    for s in sorted(all_states, key=lambda x:str(x)):
        label = mapping[s] if mapping and s in mapping else str(s)
        shape = 'doublecircle' if (s==accept) or (isinstance(accept,(set,frozenset)) and s in accept) else 'circle'
        g.node(label,label=label,shape=shape,width='1.2',height='1.2',fixedsize='true')

    trans_dict = defaultdict(list)
    if all(isinstance(k, tuple) for k in transitions.keys()):
        for (src,sym),dst in transitions.items():
            trans_dict[(src,dst)].append(str(sym))
    else:
        for src,lst in transitions.items():
            for pair in lst:
                if len(pair)==2:
                    sym,dst = pair
                    trans_dict[(src,dst)].append(EPS if sym==EPS else str(sym))
    for (src,dst),syms in trans_dict.items():
        src_label = mapping[src] if mapping and src in mapping else str(src)
        dst_label = mapping[dst] if mapping and dst in mapping else str(dst)
        g.edge(src_label,dst_label,label=",".join(sorted(syms)))
    return g

# ---------- Streamlit ----------
st.set_page_config(page_title="Algorithme de Thompson", layout="wide")
col_logo,col_title = st.columns([1,5])
with col_logo:
    try: st.image("logo.png", width=100)
    except: pass
with col_title:
    st.title("Algorithme de Thompson")
    st.caption("Construction NFA → DFA → DFA minimisé (multi-symbole)")

st.header("Entrée")
regex = st.text_input("Expression régulière", value="(a|b)*ab(a|b)*")

colA,colB,colC=st.columns([1,1,1])
with colA: build=st.button("Construire l'automate")
with colB: show_dfa=st.checkbox("Afficher DFA", value=True)
with colC: show_min=st.checkbox("Afficher DFA minimisé", value=False)

st.divider()
if 'steps' not in st.session_state: st.session_state.steps=[]
if 'final_nfa' not in st.session_state: st.session_state.final_nfa=None
if 'idx' not in st.session_state: st.session_state.idx=0

col1,col2=st.columns([1,2])
postfix_box=col1.empty()
info_box=col1.empty()
graph_box=col2.empty()
nav1,nav2=col1.columns([1,1])
prev_btn=nav1.button("← Étape précédente")
next_btn=nav2.button("Étape suivante")

# ---------- Construction ----------
if build:
    try:
        regex_expanded=expand_plus(regex.strip())
        regex2=insert_concat(regex_expanded)
        postfix=to_postfix(regex2)
        steps,final_nfa=thompson_with_steps(postfix)
        st.session_state.steps=steps
        st.session_state.final_nfa=final_nfa
        st.session_state.idx=0
        st.success("Automate NFA construit.")
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.session_state.steps=[]
        st.session_state.final_nfa=None

# ---------- Affichage NFA ----------
if st.session_state.steps:
    if next_btn and st.session_state.idx<len(st.session_state.steps)-1: st.session_state.idx+=1
    if prev_btn and st.session_state.idx>0: st.session_state.idx-=1
    idx=st.session_state.idx
    step=st.session_state.steps[idx]
    postfix_box.markdown(f"**Étape {idx+1}/{len(st.session_state.steps)} — Symbole traité :** {step['tok']}")
    info_box.write(f"Pile : {step['stack']}")
    dot_nfa=build_graph_multisym({'start':st.session_state.final_nfa['start'],
                                  'accept':st.session_state.final_nfa['accept'],
                                  'transitions':step['transitions']})
    graph_box.graphviz_chart(dot_nfa.source)

# ---------- Affichage DFA ----------
if st.session_state.final_nfa:
    dfa=nfa_to_dfa(st.session_state.final_nfa)
    if show_dfa:
        st.subheader("DFA correspondant")
        g_dfa=build_graph_multisym(dfa)
        st.graphviz_chart(g_dfa.source)

    if show_min:
        min_dfa=minimize_dfa(dfa)
        st.subheader("DFA minimisé (états I, II, III...)")
        g_min=build_graph_multisym(min_dfa
