# ia_peyton.py
import random
import datetime
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class PeytonIA:
    def __init__(self):
        self.nome = "Peyton"
        self.versao = "1.0"
        self.personalidade = {
            "tom": "amigável e profissional",
            "especialidade": "assistente multifuncional",
            "traços": ["curiosa", "prestativa", "analítica", "criativa"]
        }
        
        self.conhecimentos = {
            "saudacoes": [
                "Olá! Como posso ajudar?", 
                "Oi! Em que posso ser útil?", 
                "Hey! Pronto para conversar?",
                "Olá! Sou a Peyton, sua assistente IA!"
            ],
            "despedidas": [
                "Até logo! Foi um prazer ajudar!",
                "Tchau! Volte sempre!",
                "Até mais! Estarei aqui quando precisar!",
                "Foi ótimo conversar! Até a próxima! 👋"
            ],
            "habilidades": [
                "📊 Análise de dados e informações",
                "❓ Resposta a perguntas diversas",
                "📝 Organização e planejamento",
                "💡 Sugestões criativas",
                "🔍 Pesquisa e análise"
            ],
            "sobre": [
                "Sou a Peyton, uma IA criada para ajudar e conversar!",
                "Meu objetivo é tornar suas tarefas mais fáceis!",
                "Fui programada para aprender e me adaptar!",
                "Estou aqui para o que precisar! 🤖"
            ]
        }
        
        self.historico = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classificador = MultinomialNB()
        self.ml_treinado = False
        
        # Dados básicos para treinamento inicial
        self.dados_treino_ml = [
            "olá", "oi", "hey", "bom dia", "boa tarde", "boa noite",
            "tchau", "adeus", "até logo", "sair", "encerrar",
            "como você está", "tudo bem", "qual seu nome",
            "o que você faz", "suas habilidades", "o que sabe fazer",
            "obrigado", "thanks", "valeu", "agradeço",
            "ajuda", "help", "socorro", "preciso de ajuda"
        ]
        
        self.rotulos_treino_ml = [
            "saudacao", "saudacao", "saudacao", "saudacao", "saudacao", "saudacao",
            "despedida", "despedida", "despedida", "despedida", "despedida",
            "estado", "estado", "sobre",
            "habilidades", "habilidades", "habilidades",
            "agradecimento", "agradecimento", "agradecimento", "agradecimento",
            "ajuda", "ajuda", "ajuda", "ajuda"
        ]
        
        self.treinar_ml_basico()
    
    def treinar_ml_basico(self):
        """Treina o modelo de machine learning básico"""
        try:
            X = self.vectorizer.fit_transform(self.dados_treino_ml)
            self.classificador.fit(X, self.rotulos_treino_ml)
            self.ml_treinado = True
            print(f"✅ {self.nome}: Modelo ML treinado com sucesso!")
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
    
    def prever_intencao(self, texto):
        """Prevê a intenção do usuário usando ML"""
        if not self.ml_treinado:
            return "desconhecido"
        
        try:
            X = self.vectorizer.transform([texto.lower()])
            return self.classificador.predict(X)[0]
        except:
            return "desconhecido"
    
    def responder(self, mensagem):
        """Processa a mensagem e retorna uma resposta"""
        mensagem_limpa = mensagem.lower().strip()
        
        # Registrar entrada do usuário
        self.registrar_interacao(mensagem, "")
        
        # Prever intenção usando ML
        intencao = self.prever_intencao(mensagem_limpa)
        
        # Gerar resposta baseada na intenção
        resposta = self.gerar_resposta_inteligente(mensagem_limpa, intencao)
        
        # Atualizar histórico com a resposta
        self.historico[-1]["saida"] = resposta
        self.historico[-1]["intencao"] = intencao
        
        return resposta
    
    def gerar_resposta_inteligente(self, mensagem, intencao):
        """Gera resposta baseada na intenção detectada"""
        
        if intencao == "saudacao":
            return random.choice(self.conhecimentos["saudacoes"])
        
        elif intencao == "despedida":
            return random.choice(self.conhecimentos["despedidas"])
        
        elif intencao == "sobre":
            return random.choice(self.conhecimentos["sobre"])
        
        elif intencao == "habilidades":
            habilidades = "\n".join(self.conhecimentos["habilidades"])
            return f"Minhas habilidades incluem:\n{habilidades}"
        
        elif intencao == "agradecimento":
            return "De nada! Fico feliz em ajudar! 😊"
        
        elif intencao == "estado":
            return "Estou funcionando perfeitamente! Pronta para ajudar!"
        
        elif intencao == "ajuda":
            return "Claro! Me diga exatamente com o que você precisa de ajuda."
        
        else:
            # Resposta para mensagens não reconhecidas
            return self.gerar_resposta_criativa(mensagem)
    
    def gerar_resposta_criativa(self, mensagem):
        """Gera respostas criativas para mensagens não reconhecidas"""
        
        respostas_padrao = [
            f"Interessante! Você disse: '{mensagem}'. Como posso ajudar com isso?",
            f"Hmm, '{mensagem}'... Pode me dar mais detalhes?",
            f"Entendi que você mencionou: '{mensagem}'. O que gostaria de saber sobre isso?",
            f"Sobre '{mensagem}', posso ajudar de alguma forma específica?",
            f"Curioso! '{mensagem}' é um tópico interessante. Como posso ser útil?"
        ]
        
        # Análise simples de palavras-chave
        palavras_chave = {
            "hora": f"Agora são {datetime.datetime.now().strftime('%H:%M')}",
            "data": f"Hoje é {datetime.datetime.now().strftime('%d/%m/%Y')}",
            "nome": f"Meu nome é {self.nome}! Prazer em conhecê-lo!",
            "idade": "Como IA, não tenho idade, mas fui criada recentemente!",
            "clima": "Não tenho acesso ao clima em tempo real, mas posso ajudar com outras coisas!",
            "piada": self.gerar_piada()
        }
        
        for palavra, resposta in palavras_chave.items():
            if palavra in mensagem:
                return resposta
        
        return random.choice(respostas_padrao)
    
    def gerar_piada(self):
        """Gera uma piada aleatória"""
        piadas = [
            "Por que o Python foi para a terapia? Porque tinha muitos issues! 🐍",
            "Qual é o café favorito do desenvolvedor? Java! ☕",
            "Por que os elétrons nunca são presos? Porque eles sempre têm um álibi! ⚡",
            "Quantos programadores são necessários para trocar uma lâmpada? Nenhum, é um problema de hardware! 💡"
        ]
        return random.choice(piadas)
    
    def registrar_interacao(self, entrada, saida):
        """Registra a interação no histórico"""
        timestamp = datetime.datetime.now()
        self.historico.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "entrada": entrada,
            "saida": saida,
            "intencao": "pendente"
        })
    
    def mostrar_historico(self):
        """Mostra o histórico de conversas"""
        if not self.historico:
            return "Nenhuma conversa registrada ainda."
        
        historico_str = f"📋 Histórico da {self.nome}:\n"
        for i, interacao in enumerate(self.historico[-10:], 1):  # Últimas 10
            historico_str += f"\n{i}. [{interacao['timestamp']}]\n"
            historico_str += f"   Você: {interacao['entrada']}\n"
            historico_str += f"   {self.nome}: {interacao['saida']}\n"
        
        return historico_str
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas de uso"""
        total_interacoes = len(self.historico)
        intencoes = [interacao.get('intencao', 'desconhecido') for interacao in self.historico]
        
        estatisticas = {
            "Total de interações": total_interacoes,
            "Primeira interação": self.historico[0]['timestamp'] if self.historico else "N/A",
            "Intenções detectadas": dict(zip(*np.unique(intencoes, return_counts=True)))
        }
        
        return estatisticas
    
    def salvar_historico(self, arquivo="historico_peyton.json"):
        """Salva o histórico em arquivo JSON"""
        try:
            with open(arquivo, 'w', encoding='utf-8') as f:
                json.dump(self.historico, f, ensure_ascii=False, indent=2)
            return f"Histórico salvo em {arquivo}!"
        except Exception as e:
            return f"Erro ao salvar histórico: {e}"
    
    def carregar_historico(self, arquivo="historico_peyton.json"):
        """Carrega histórico de arquivo JSON"""
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                self.historico = json.load(f)
            return f"Histórico carregado de {arquivo}!"
        except FileNotFoundError:
            return "Arquivo de histórico não encontrado."
        except Exception as e:
            return f"Erro ao carregar histórico: {e}"

def main():
    """Função principal para executar a IA Peyton"""
    peyton = PeytonIA()
    
    print(f"🤖 === IA {peyton.nome} v{peyton.versao} Ativada ===")
    print("💬 Digite sua mensagem (ou 'ajuda' para comandos especiais)")
    print("❌ Digite 'sair' para encerrar\n")
    
    while True:
        try:
            usuario_input = input("👤 Você: ").strip()
            
            if not usuario_input:
                continue
            
            # Comandos especiais
            if usuario_input.lower() in ['sair', 'quit', 'exit', 'bye']:
                print(f"🤖 {peyton.nome}: {random.choice(peyton.conhecimentos['despedidas'])}")
                break
            
            elif usuario_input.lower() == 'historico':
                print(f"\n{peyton.mostrar_historico()}")
                continue
            
            elif usuario_input.lower() == 'estatisticas':
                stats = peyton.mostrar_estatisticas()
                print(f"\n📊 Estatísticas:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            elif usuario_input.lower() == 'salvar':
                resultado = peyton.salvar_historico()
                print(f"🤖 {peyton.nome}: {resultado}")
                continue
            
            elif usuario_input.lower() == 'carregar':
                resultado = peyton.carregar_historico()
                print(f"🤖 {peyton.nome}: {resultado}")
                continue
            
            elif usuario_input.lower() == 'limpar':
                peyton.historico = []
                print(f"🤖 {peyton.nome}: Histórico limpo!")
                continue
            
            elif usuario_input.lower() == 'ajuda':
                print(f"""
🤖 {peyton.nome} - Comandos Disponíveis:
• 'historico' - Mostra últimas conversas
• 'estatisticas' - Mostra estatísticas de uso
• 'salvar' - Salva histórico em arquivo
• 'carregar' - Carrega histórico de arquivo
• 'limpar' - Limpa o histórico
• 'sair' - Encerra a conversa
• Ou simplesmente converse normalmente!
                """)
                continue
            
            # Processar mensagem normal
            resposta = peyton.responder(usuario_input)
            print(f"🤖 {peyton.nome}: {resposta}")
            
        except KeyboardInterrupt:
            print(f"\n🤖 {peyton.nome}: Encerrando conversa... Até mais! 👋")
            break
        except Exception as e:
            print(f"🤖 {peyton.nome}: Ops, algo deu errado! Erro: {e}")

if __name__ == "__main__":
    main()
