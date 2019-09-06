import os
import re
import time
import webbrowser
from contextlib import closing
import pandas as pd
import glob
from pathlib import Path
import ntpath
from bs4 import BeautifulSoup, NavigableString, Tag
from requests import get
from requests.exceptions import RequestException


class MITREWebscraper():

    with open('mitretechniques.txt', 'r') as f:
        technique_list = f.read().splitlines()

    def simple_get(self, url):
        """
        Attempts to get the content at `url` by making an HTTP GET request.
        If the content-type of response is some kind of HTML/XML, return the
        text content, otherwise return None.
        """
        try:
            with closing(get(url, stream=True)) as resp:
                if self.is_good_response(resp):
                    return resp.content
                else:
                    return None

        except RequestException as e:
            self.log_error('Error during requests to {0} : {1}'.format(url, str(e)))
            return None


    def is_good_response(self, resp):
        """
        Returns True if the response seems to be HTML, False otherwise.
        """
        content_type = resp.headers['Content-Type'].lower()
        return (resp.status_code == 200 
                and content_type is not None 
                and content_type.find('html') > -1)


    def log_error(self, e):
        """
        It is always a good idea to log errors. 
        This function just prints them, but you can
        make it do anything.
        """
        print(e)
        
    def get_technique_description(self, html):
        """
        Function that gets the description of Mitre techniques
        """
        article_text = ''
        article = html.find("div", {"class":"col-md-8 description-body"}).findAll('p')
        for element in article:
            article_text += " " + ''.join(element.findAll(text = True))

        result = re.sub(r"\[.*?\]", '', article_text)    
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def get_technique_response(self, html, m_type):
        """
        Function that gets the mitigation methods for a technique
        
        m_type (str) => the Detection or Mitigation section of MITRE, pass either `detection` or `mitigation`
        """
        article_text = ''
        article = html.find("h2", {"id": m_type})
        
        nextNode = article
        
        if nextNode is None:
            return article_text
        
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, Tag):
                if nextNode.name == "h2":
                    break
                article_text += (nextNode.get_text(strip=True).strip())
                
        result = re.sub(r"\[.*?\]", '', article_text)    
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def get_technique(self, html):
        """
        Function that gets the technique of the attack
        """    
        
        return BeautifulSoup(html, features="lxml").find("h1").text.strip()

    def get_tactics(self, html):
        """
        Function that gets the tactic of the attack
        """
        
        article = BeautifulSoup(html, features="lxml").find_all("div", {"class":"card-data"})
        for dom in article:
            if dom.find("span", {"class": "card-title"}).text.strip() == 'Tactic':
                tactics = dom.text.strip('')[8:].replace(' ','')
        
        return tactics

    def generate_data(self, *args, update=False):
        """
        *args (str) => can only be any combination of `description`, `detection`, `mitigation` or `all`
        """
        
        func_args = args
        update_eval = False

        if "all" in args:
            func_args = ['description', 'detection', 'mitigation']
        
        for ind,item in enumerate(self.technique_list):
            if update:
                self._download_data(ind, item, func_args)
            else:
                #Check if the data for this technique has been downloaded
                for option in func_args:
                    update_eval = os.path.exists(f'{option}/{item}.txt')

                if not update_eval:
                    self._download_data(ind, item, func_args)
                else:
                    print(f'Already have technique ID {item} ... {ind+1}/{len(self.technique_list)}')        


    def _download_data(self, ind, item, func_args):

        print(f'Getting technique ID {item} ... {ind+1}/{len(self.technique_list)}')        
        raw_html = self.simple_get(f'https://attack.mitre.org/techniques/{item}/')
        html = BeautifulSoup(raw_html, 'html.parser')

        for option in func_args:
            if option == "description":
                desc = self.get_technique_description(html)
            else:
                desc = self.get_technique_response(html, option)

            self._write_data(item, option, desc)

        time.sleep(30)

            
    def _write_data(self, t_id, m_type, desc):
        """
        Writes Mitre Tactic info to a file in the specified folders.
        
        t_id (str) => Mitre technique ID
        m_type (str) => Mitre information header that acts as the folder name (Description, Detection, Technique)
        desc (str) => Description, Mitigation or Detection explanation    
        """
        
        if desc == '':
            return
        
        if not os.path.exists(f'{m_type}/'):
            os.makedirs(f'{m_type}/')
            
        with open(f'{m_type}/{t_id}.txt', 'w') as f:
            f.write(desc)
        
    def data_scope_validation(self, m_type, manual=False):
        
        all_techniques = set(self.technique_list)
        curr_files = set(map(lambda x: x[:-4], os.listdir(f'{m_type}/')))
        verify_list = all_techniques - curr_files
        
        curr_coverage_score = len(curr_files)
        total_coverage_score = len(all_techniques)
        
        if len(verify_list) != 0:
            
            for item in verify_list:
                print(f"Validating for you that {m_type} does not exist on the missing technique page {item}.")
                raw_html = self.simple_get(f'https://attack.mitre.org/techniques/{item}/')
                html = BeautifulSoup(raw_html, 'html.parser')

                article = html.find("h2", {"id": m_type})
                if article != '':
                    pass
                else:
                    total_coverage_score -= 1
                
        print("You have the {} MITRE data for {}% of techniques.".format(
                                                                        m_type,
                                                                        round(curr_coverage_score/total_coverage_score, 2) * 100
                                                                        ))
        
        if manual:
            print("Opening URL pages for manual validation...")        
            for item in verify_list:
                webbrowser.get('chrome').open_new_tab(f'https://attack.mitre.org/techniques/{item}/')
                
                
                
    def path_leaf(self,path):

        '''
        Gives the path
        '''
        ntpath.basename("a/b/c")
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


    def document_list(self,f_path):
        '''
        Gives the list of technique numbers

        '''

        doc_lst=[self.path_leaf(i) for i in f_path]
        doc_new_lst=[]
        for doc in doc_lst:
            doc_name=re.sub(r'[.txt]','',doc)
            doc_new_lst.append(doc_name)
        return doc_new_lst

    def tech_name(self,file):   

        '''
         Gives the list of technique names
        '''

        tech_lst=[]
        for name in file:
            website=self.simple_get('https://attack.mitre.org/techniques/'+name)
            tech=self.get_technique(website)
            tech_lst.append(tech)
        return tech_lst

    def tactic_name(self,file):  
        '''
        Gives a list of tactic names
        '''
        tactic_lst=[]
        for name in file:
            website=self.simple_get('https://attack.mitre.org/techniques/'+name)
            tactic=self.get_tactics(website)
            tactic_lst.append(tactic)
        return tactic_lst

    

    def description(self,file_path):

        '''
        Gives a list of description for the file_path passed
        '''

        description_lst=[]
        for file_path in file_path:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_data = file.read()
                description_lst.append(file_data)
        return description_lst

