import React from "react";

class SearchBar extends React.Component {
  state = { term: "machine learning", way: "tfidf" };

  onFormSubmit = event => {
    event.preventDefault();
    this.props.onFormSubmit(this.state.term, this.state.way);
  };

  render() {
    return (
      <div className="ui segment">
        <form className="ui form" onSubmit={this.onFormSubmit}>
          <div className="ui fluid action input left icon">
            <i className="search icon"></i>
            <textarea
              type="text"
              placeholder="Search..."
              onChange={e => this.setState({ term: e.target.value })}
              rows="4"
            >{this.state.term}</textarea>
            <div>
              <select
                className="ui selection dropdown"
                onChange={e => this.setState({ way: e.target.value })}
              >
                <option value="tfidf">TF-IDF</option>
                <option value="bm25">BM25</option>
                <option value="bert">BERT</option>
              </select>
            </div>
          </div>
        </form>
      </div>
    );
  }
}

export default SearchBar;
